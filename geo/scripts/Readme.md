# Using tomopy to preprocess and reconstruct uCT data

tom_recon.py is the main script that calls tomopy functions and does the preprocessing with
subsequent 3D reconstruction.

It has two command line arguments that allow splitting the full dataset of 2D
projections into subsets.

./tom_recon.py nstep noffset

Here 'nstep' is the step size when going through the array of projections.
With nstep=1 all projections are used for reconstruction, with nstep=2 every other projection is
used for reconstruction, etc.

'noffset' is the offset at which this selection starts.

For instance the pair of reconstruction runs

./tom_recon.py 2 0

and

./tom_recon.py 2 1

will create two reconstructions each based on an independent subset of
projections. These can be then used as a basis for training the noise2noise
network

I have hard-coded the base name of the input data files "Bent2_under_op__E",
but it is easy for you to fixt that.
For this initial study, I stored the four required input files in
/gpfs/alpine/gen011/scratch/l2v/tomopy_run/data/bentonite
Just copy them where you need them.

At present I use tomopy-1.4.2, which I installed in anaconda distribution with
rapids.

export PATH=/ccs/home/l2v/nv_rapids_0.9_gcc_6.4.0/anaconda3/bin:$PATH

