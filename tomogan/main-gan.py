#! /home/zliu0/usr/anaconda3/envs/tfgpu/bin/python
import tensorflow as tf 
tf.enable_eager_execution()
import numpy as np 
from util import save2img
import sys, os, time, argparse, shutil, scipy, h5py, glob
from models import tomogan_disc as make_discriminator_model  # import a disc model
from models import unet as make_generator_model           # import a generator model
from data_processor import bkgdGen, gen_train_batch_bg, get1batch4test
import horovod.tensorflow as hvd 

tf.logging.set_verbosity(tf.logging.ERROR)

parser = argparse.ArgumentParser(description='encode sinogram image.')
parser.add_argument('-gpus',  type=int, default=6, help='number of GPUs')
parser.add_argument('-expName', type=str, required=True, help='Experiment name')
parser.add_argument('-lmse', type=float, default=0.5, help='lambda mse')
parser.add_argument('-lperc', type=float, default=2.0, help='lambda perceptual')
parser.add_argument('-ladv', type=float, default=20, help='lambda adv')
parser.add_argument('-lunet', type=int, default=3, help='Unet layers')
parser.add_argument('-depth', type=int, default=3, help='input depth')
parser.add_argument('-itg', type=int, default=1, help='iterations for G')
parser.add_argument('-itd', type=int, default=2, help='iterations for D')
parser.add_argument('-xtrain', type=str, required=True, help='file name of X for training')
parser.add_argument('-ytrain', type=str, required=True, help='file name of Y for training')
parser.add_argument('-xtest', type=str, required=True, help='file name of X for testing')
parser.add_argument('-ytest', type=str, required=True, help='file name of Y for testing')

args, unparsed = parser.parse_known_args()

hvd.init()

os.environ['CUDA_VISIBLE_DEVICES'] = str(hvd.local_rank())
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # disable printing INFO, WARNING, and ERROR

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.99
#config.gpu_options.visible_device_list = str(hvd.local_rank())
sess = tf.Session(config = config)
tf.keras.backend.set_session(sess)

mb_size = 4
img_size = 512
in_depth = args.depth
disc_iters, gene_iters = args.itd, args.itg
lambda_mse, lambda_adv, lambda_perc = args.lmse, args.ladv, args.lperc

if hvd.rank() == 0:
    itr_out_dir = args.expName + '-itrOut'
    if os.path.isdir(itr_out_dir): 
        shutil.rmtree(itr_out_dir)
    os.mkdir(itr_out_dir) # to save temp output

    # redirect print to a file
    sys.stdout = open('%s/%s' % (itr_out_dir, 'iter-prints.log'), 'w') 

    print('X train: {}\nY train: {}\nX test: {}\nY test: {}'.format(args.xtrain, args.ytrain, args.xtest, args.ytest))

# build minibatch data generator with prefetch
mb_data_iter = bkgdGen(data_generator=gen_train_batch_bg(x_fn=args.xtrain, \
                                      y_fn=args.ytrain, mb_size=mb_size, \
                                      in_depth=in_depth, img_size=img_size), \
                       max_prefetch=16)   

generator = make_generator_model(input_shape=(None, None, in_depth), nlayers=args.lunet ) 
discriminator = make_discriminator_model(input_shape=(img_size, img_size, 1))

# input range should be [0, 255]
feature_extractor_vgg = tf.keras.applications.VGG19(\
                        weights='vgg19_weights_notop.h5', \
                        include_top=False)

# This method returns a helper function to compute cross entropy loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def adversarial_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


gen_optimizer  = tf.train.AdamOptimizer(1e-4*hvd.size())
disc_optimizer = tf.train.AdamOptimizer(1e-4*hvd.size())

global_step = tf.train.get_or_create_global_step()

if hvd.rank() == 0:
    ckpt_dir = './%s_checkpoints'%(args.expName)
    ckpt_prefix = os.path.join(ckpt_dir, "ckpt")
    ckpt = tf.train.Checkpoint(generator_optimizer=gen_optimizer,
                            discriminator_optimizer=disc_optimizer,
                            generator=generator,
                            discriminator=discriminator,
                            step_counter=global_step)
    print("loading checkpoint %s"%tf.train.latest_checkpoint(ckpt_dir))
    ckpt.restore(tf.train.latest_checkpoint(ckpt_dir))

hvd.broadcast_global_variables(0)

for epoch in range(40001):
    time_git_st = time.time()
    for _ge in range(gene_iters):
        X_mb, y_mb = mb_data_iter.next() # with prefetch
        with tf.GradientTape() as gen_tape:
            gen_tape.watch(generator.trainable_variables)

            gen_imgs = generator(X_mb, training=True)
            disc_fake_o = discriminator(gen_imgs, training=False)

            loss_mse = tf.losses.mean_squared_error(gen_imgs, y_mb)
            loss_adv = adversarial_loss(disc_fake_o)

            vggf_gt  = feature_extractor_vgg.predict(tf.concat([y_mb, y_mb, y_mb], 3).numpy())
            vggf_gen = feature_extractor_vgg.predict(tf.concat([gen_imgs, gen_imgs, gen_imgs], 3).numpy())
            perc_loss= tf.losses.mean_squared_error(vggf_gt.reshape(-1), vggf_gen.reshape(-1))

            gen_loss = lambda_adv * loss_adv + lambda_mse * loss_mse + lambda_perc * perc_loss

        gen_tape = hvd.DistributedGradientTape(gen_tape)
        gen_gradients = gen_tape.gradient(gen_loss, generator.trainable_variables)
        gen_optimizer.apply_gradients(zip(gen_gradients, generator.trainable_variables),
                                      global_step=tf.train.get_or_create_global_step())
        if _ge == 0:
            hvd.broadcast_variables(generator.trainable_variables, root_rank=0)
            hvd.broadcast_variables(gen_optimizer.variables(), root_rank=0)
    if hvd.rank() == 0:
        itr_prints_gen = '[Info] Epoch: %05d, gloss: %.2f (mse%.3f, adv%.3f, perc:%.3f), gen_elapse: %.2fs/itr' % (\
                     epoch, gen_loss, loss_mse*lambda_mse, loss_adv*lambda_adv, perc_loss*lambda_perc, \
                     (time.time() - time_git_st)/gene_iters, )
    time_dit_st = time.time()

    for _de in range(disc_iters):
        X_mb, y_mb = mb_data_iter.next() # with prefetch        
        with tf.GradientTape() as disc_tape:
            disc_tape.watch(discriminator.trainable_variables)

            gen_imgs = generator(X_mb, training=False)

            disc_real_o = discriminator(y_mb, training=True)
            disc_fake_o = discriminator(gen_imgs, training=True)

            disc_loss = discriminator_loss(disc_real_o, disc_fake_o)

        disc_tape = hvd.DistributedGradientTape(disc_tape)    
        disc_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
        disc_optimizer.apply_gradients(zip(disc_gradients, discriminator.trainable_variables),
                                           global_step=tf.train.get_or_create_global_step())
        if _de == 0:
            hvd.broadcast_variables(discriminator.trainable_variables, root_rank=0)
            hvd.broadcast_variables(disc_optimizer.variables(), root_rank=0)
    if hvd.rank() == 0:
        print('%s; dloss: %.2f (r%.3f, f%.3f), disc_elapse: %.2fs/itr, gan_elapse: %.2fs/itr' % (itr_prints_gen,\
          disc_loss, disc_real_o.numpy().mean(), disc_fake_o.numpy().mean(), \
          (time.time() - time_dit_st)/disc_iters, time.time()-time_git_st))

    if epoch % (100//gene_iters) == 0:
        X222, y222 = get1batch4test(x_fn=args.xtest, y_fn=args.ytest, in_depth=in_depth)
        pred_img = generator.predict(X222[:1])
        if hvd.rank() == 0:
            save2img(pred_img[0,:,:,0], '%s/it%05d.png' % (itr_out_dir, epoch))
            if epoch == 0: 
                save2img(y222[0,:,:,0], '%s/gtruth.png' % (itr_out_dir))
                save2img(X222[0,:,:,in_depth//2], '%s/noisy.png' % (itr_out_dir))
        if hvd.rank() == 0: 
            generator.save("%s/%s-it%05d.h5" % (itr_out_dir, args.expName, epoch), \
                       include_optimizer=False)

        # discriminator.save("%s/disc-it%05d.h5" % (itr_out_dir, epoch), \
        #                include_optimizer=False)
        if hvd.rank() == 0: 
            #saving checkpoint
            ckpt.save(file_prefix = ckpt_prefix)
    sys.stdout.flush()

