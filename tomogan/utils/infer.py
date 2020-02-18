import tensorflow as tf
import h5py
import numpy as np
from util import save2img
from tensorflow.keras import backend as K
import argparse
import time 

args = argparse.ArgumentParser()
args.add_argument('--data', default=None, type=str,
                  help='low fedility data')
args.add_argument('--model', default=None, type=str,
                  help='saved model h5 file')
args.add_argument('--batch-size', default=4, type=int,
                  help='batch size for inferences')
args.add_argument('--depth', default=3, type=int,
                  help='trained model input depth')
args.add_argument('--memory-ratio', default=0.99, type=float,
                  help='number bigger than 1 will use UVM')
args = args.parse_args()
data= args.data
model = args.model
batch_size = args.batch_size 
in_depth = args.depth
ratio = args.memory_ratio

gpuOptions = tf.GPUOptions(per_process_gpu_memory_fraction=ratio)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpuOptions))
K.set_session(sess)


generator = tf.keras.models.load_model(model)
X_pred = []

with h5py.File(data, 'r') as h5:
    X = h5['images'][:].astype(np.float32)

start = time.time()
for idx in range(0, X.shape[0], batch_size):
    batch = []
    for s_idx in range(idx, idx+batch_size):
        if s_idx+in_depth >= X.shape[0]:
            batch.append(np.transpose(X[s_idx : (s_idx-in_depth) : -1], (1, 2, 0)))
        else:    
            batch.append(np.transpose(X[s_idx : (s_idx+in_depth)], (1, 2, 0)))
    X_pred.append(generator.predict(np.array(batch)))

print('memory ratio: %g runtime: %d(s)'%(ratio, time.time()-start))

X_pred = np.reshape(np.array(X_pred), X.shape)
with h5py.File('GAN_reconstructed.h5', 'w') as h5:
    h5['images'] = X_pred 

print(X_pred.shape)
save2img(X_pred[0,:,:], 'sample.png')

