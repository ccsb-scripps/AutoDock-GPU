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
3. NWI is not currently an input - it is currently set only to 1
4. DEVICE=GPU will only use 1 GPU (i.e. on Summit the other 5 will be idle)

Points 3 and 4 are of course temporary and will be addressed ASAP
Point 2 can be addressed if needed - I was told this was the important method

A sample Summit script is included in the repository: sample_summit_jobscript
