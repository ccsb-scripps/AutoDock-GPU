#ocladock
==================

OCLADock: OpenCL Accelerated Molecular Docking

#Updated commands for open-source release

##Basic command
```zsh
./ocladock -ffile <protein>.maps.fld -lfile <ligand>.pdbqt -nrun <nruns>
```

##Example

```zsh
./ocladock_64wi -ffile ./input_data/1stp/derived/1stp_protein.maps.fld -lfile ./input_data/1stp/derived/1stp_ligand.pdbqt -nrun 10
```

##Supported arguments
For a complete list of available arguments and their default values, check: `host/src/getparameters.cpp`.

| Argument | Description                  | Default value |
|----------|------------------------------|---------------|
| -nrun    | # Docking runs               | 1             |
| -nev     | # Energy evaluations         | 2500000       |
| -ngen    | # Generations                | 27000         |
| -lsit    | # Local-search iterations (max.) | 300       |
| -psize   | Population size              | 150           |
| -mrat    | Mutation rate                | 2 (%)         |
| -crat    | Crossover rate               | 80 (%)        |
| -lsrat   | Local-search rate            | 6 (%)         |
| -trat    | Tournament rate              | 60 (%)        |
| -resname | Name for docking output log  | "docking"     |
