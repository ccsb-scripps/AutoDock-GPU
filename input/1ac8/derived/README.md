How to run:
```sh
autodock_gpu -M 1ac8_protein.maps.fld -L 1ac8_ligand.pdbqt
```

Example of expected output. Docking is stochastic, numbers will vary.

```
 
AutoDock-GPU version: v1.5.3-45-gac642592fc4537bc66dcaecd6aef0754c68c5cf8-dirty

Running 1 docking calculation

Kernel source used for development:      ./device/calcenergy.cl                  
Kernel string used for building:         ./host/inc/stringify.h                  
Kernel compilation flags:                 -I ./device -I ./common -DN128WI   -cl-mad-enable
OpenCL device:                           NVIDIA GeForce GTX 1080
(Thread 0 is setting up Job #1)

Running Job #1
    Using heuristics: (capped) number of evaluations set to 1132076
    Local-search chosen method is: ADADELTA (ad)

Executing docking runs, stopping automatically after either reaching 0.15 kcal/mol standard deviation of
the best molecules of the last 4 * 5 generations, 42000 generations, or 1132076 evaluations:

Generations |  Evaluations |     Threshold    |  Average energy of best 10%  | Samples |    Best energy
------------+--------------+------------------+------------------------------+---------+-------------------
          0 |          150 |   -1.12 kcal/mol |   -2.17 +/-    0.28 kcal/mol |       4 |   -2.56 kcal/mol
          5 |        64488 |   -1.12 kcal/mol |   -3.74 +/-    1.16 kcal/mol |     838 |   -5.70 kcal/mol
         10 |       129266 |   -3.71 kcal/mol |   -5.65 +/-    0.07 kcal/mol |     146 |   -5.70 kcal/mol
         15 |       198130 |   -5.63 kcal/mol |   -5.68 +/-    0.01 kcal/mol |     113 |   -5.70 kcal/mol
         20 |       266458 |   -5.66 kcal/mol |   -5.68 +/-    0.01 kcal/mol |     155 |   -5.70 kcal/mol
         25 |       335892 |   -5.67 kcal/mol |   -5.68 +/-    0.01 kcal/mol |     166 |   -5.70 kcal/mol
------------+--------------+------------------+------------------------------+---------+-------------------

                                   Finished evaluation after reaching
                                   -5.67 +/-    0.04 kcal/mol combined.
                               580 samples, best energy    -5.70 kcal/mol.


Job #1 took 0.347 sec after waiting 0.181 sec for setup

(Thread 0 is processing Job #1)
Run time of entire job set (1 file): 0.546 sec
Processing time: 0.017 sec

All jobs ran without errors.
``` 
