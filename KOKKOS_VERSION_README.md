To run on Summit:

Load modules:
module load pgi/19.9
module load cuda/10.1.243

PGI is used because I already had a working version of Kokkos on Summit installed with PGI.

Compiling is the same as before, e.g.:

make DEVICE=GPU

Changes:
1. DEVICE=CPU uses OpenMP; to disable, use DEVICE=SERIAL
2. ADADELTA is the ONLY option right now
3. NWI has changed to NUM_OF_THREADS_PER_BLOCK. Specify as desired; the default is 32 on GPU and 1 on CPU

A sample Summit script is included in the repository: sample_summit_jobscript
