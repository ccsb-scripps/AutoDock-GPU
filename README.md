AutoDock-GPU: AutoDock for GPUs and other accelerators
======================================================

<img src="logo.png" width="200">

# About

* OpenCL and Cuda accelerated version of AutoDock4.2.6. It leverages its embarrasingly parallelizable LGA by processing ligand-receptor poses in parallel over multiple compute units.
* The OpenCL version was developed in collaboration with TU-Darmstadt and is able to target CPU, GPU, and FPGA architectures.
* The Cuda version was developed in collaboration with Nvidia to run AutoDock-GPU on the Oak Ridge National Laboratory's (ORNL) Summit, and it included a batched ligand pipeline developed by Aaron Scheinberg from Jubilee Development.

# Citation

Accelerating AutoDock4 with GPUs and Gradient-Based Local Search, [J. Chem. Theory Comput.](https://doi.org/10.1021/acs.jctc.0c01006) 2021, 10.1021/acs.jctc.0c01006

See [more relevant papers](https://github.com/ccsb-scripps/AutoDock-GPU/wiki/Publications)

# Features

* Gradient-based local search methods (e.g. ADADELTA), as well as an improved version of Solis-Wets from AutoDock 4.
* It targets platforms based on GPU as well as multicore CPU accelerators.
* Observed speedups of up to 4x (quad-core CPU) and 56x (GPU) over the original serial AutoDock 4.2 (Solis-Wets) on CPU. The Cuda version is currently even faster than the OpenCL version.
* A batched ligand pipeline to run virtual screenings on the same receptor (both OpenCL and Cuda)

# Setup

| Operating system                         | CPU                          | GPU                                            |
|:----------------------------------------:|:----------------------------:|:----------------------------------------------:|
|CentOS 6.7 & 6.8 / Ubuntu 14.04 & 16.04   | Intel SDK for OpenCL 2017    | AMD APP SDK v3.0 / CUDA v8.0, v9.0, and v10.0  |
|macOS Catalina 10.15.1                    | Apple / Intel                | Apple / Intel Iris, Radeon Vega 64, Radeon VII |


Other environments or configurations likely work as well, but are untested.

# Compilation

```zsh
make DEVICE=<TYPE> NUMWI=<NWI>
```

| Parameters | Description                  | Values                                             |
|:----------:|:----------------------------:|:--------------------------------------------------:|
| `<TYPE>`   | Accelerator chosen           | `CPU`, `GPU`, `CUDA`, `OCLGPU`                     |
| `<NWI>`    | work-group/thread block size | `1`, `2`, `4`, `8`, `16`, `32`, `64`, `128`, `256` |

When `DEVICE=GPU` is chosen, the Makefile will automatically tests if it can compile Cuda succesfully. To override, use `DEVICE=CUDA` or `DEVICE=OCLGPU`. The cpu target is only supported using OpenCL. Furthermore, an OpenMP-enabled overlapped pipeline (for setup and processing) can be compiled with `OVERLAP=ON`.
Hints: The best work-group size depends on the GPU and workload. Try `NUMWI=128` or `NUMWI=64` for modern cards with the example workloads. On macOS, use `NUMWI=1` for CPUs.

After successful compilation, the host binary **autodock_&lt;type&gt;_&lt;N&gt;wi** is placed under [bin](./bin).

| Binary-name portion | Description                  | Values                                            |
|:-------------------:|:----------------------------:|:-------------------------------------------------:|
| **&lt;type&gt;**    | Accelerator chosen           | `cpu`, `gpu`                                      |
| **&lt;N&gt;**       | work-group/thread block size | `1`, `2`, `4`, `8`,`16`, `32`, `64`, `128`, `256` |


# Usage

## Basic command
```zsh
./bin/autodock_<type>_<N>wi \
--ffile <protein>.maps.fld \
--lfile <ligand>.pdbqt \
--nrun <nruns>
```

| Mandatory options|   | Description   | Value                     |
|:----------------:|:-:|:-------------:|:-------------------------:|
|--ffile           |-M |Protein file   |&lt;protein&gt;.maps.fld   |
|--lfile           |-L |Ligand file    |&lt;ligand&gt;.pdbqt       |

Both options can alternatively be provided in the contents of the files specified with `--filelist (-B)` (see below for format) and `--import_dpf (-I)` (AD4 dpf file format).

## Example
```zsh
./bin/autodock_gpu_64wi \
--ffile ./input/1stp/derived/1stp_protein.maps.fld \
--lfile ./input/1stp/derived/1stp_ligand.pdbqt
```
By default the output log file is written in the current working folder. Examples of output logs can be found under [examples/output](examples/output/).

## Supported arguments

| Argument          |   | Description                                           | Default value    |
|:------------------|:-:|:------------------------------------------------------|-----------------:|
|--lfile            |-L | Ligand pdbqt file                                     | no default       |
|--ffile            |-M | Grid map files descriptor fld file                    | no default       |
|--flexres          |-F | Flexible residue pdbqt file                           | no default       |
|--filelist         |-B | Batch file                                            | no default       |
|--import_dpf       |-I | Import AD4-type dpf input file (only partial support) | no default       |
|--resnam           |-N | Name for docking output log                           | ligand basename  |
|--xraylfile        |-R | reference ligand file for RMSD analysis               | ligand file      |
|--devnum           |-D | OpenCL/Cuda device number (counting starts at 1)      | 1                |
|--derivtype        |-T | Derivative atom types (e.g. C1,C2,C3=C/S4=S/H5=HD)    | no default       |
|--modpair          |-P | Modify vdW pair params (e.g. C1:S4,1.60,1.200,13,7)   | no default       |
|--heuristics       |-H | Ligand-based automatic search method and # evals      | 1 (yes)          |
|--heurmax          |-E | Asymptotic heuristics # evals limit (smooth limit)    | 12000000         |
|--autostop         |-A | Automatic stopping criterion based on convergence     | 1 (yes)          |
|--asfreq           |-a | AutoStop testing frequency (in # of generations)      | 5                |
|--contact_analysis |-C | Perform distance-based analysis (description below)   | 0 (no)           |
|--xml2dlg          |-X | One (or many) AD-GPU xml file(s) to convert to dlg(s) | no default       |
|--xmloutput        |-x | Specify if xml output format is wanted                | 1 (yes)          |
|--loadxml          |-c | Load initial population from xml results file         | no default       |
|--dlgoutput        |-d | Control if dlg output is created                      | 1 (yes)          |
|--dlg2stdout       |-2 | Write dlg file output to stdout (if not OVERLAP=ON)   | 0 (no)           |
|--seed             |-s | Random number seeds (up to three comma-sep. integers) | time, process id |
|--ubmod            |-u | Unbound model: 0 (bound), 1 (extended), 2 (compact)   | 0 (same as bound)|
|--nrun             |-n | # LGA runs                                            | 20               |
|--nev              |-e | # Score evaluations (max.) per LGA run                | 2500000          |
|--ngen             |-g | # Generations (max.) per LGA run                      | 42000            |
|--lsmet            |-l | Local-search method                                   | ad (ADADELTA)    |
|--lsit             |-i | # Local-search iterations (max.)                      | 300              |
|--psize            |-p | Population size                                       | 150              |
|--mrat             |   | Mutation rate                                         | 2   (%)          |
|--crat             |   | Crossover rate                                        | 80  (%)          |
|--lsrat            |   | Local-search rate                                     | 100 (%)          |
|--trat             |   | Tournament (selection) rate                           | 60  (%)          |
|--rlige            |   | Print reference ligand energies                       | 0 (no)           |
|--hsym             |   | Handle symmetry in RMSD calc.                         | 1 (yes)          |
|--rmstol           |   | RMSD clustering tolerance                             | 2 (Å)            |
|--dmov             |   | Maximum LGA movement delta                            | 6 (Å)            |
|--dang             |   | Maximum LGA angle delta                               | 90 (°)           |
|--rholb            |   | Solis-Wets lower bound of rho parameter               | 0.01             |
|--lsmov            |   | Solis-Wets movement delta                             | 2 (Å)            |
|--lsang            |   | Solis-Wets angle delta                                | 75 (°)           |
|--cslim            |   | Solis-Wets cons. success/failure limit to adjust rho  | 4                |
|--smooth           |   | Smoothing parameter for vdW interactions              | 0.5 (Å)          |
|--elecmindist      |   | Min. electrostatic potential distance (w/ dpf: 0.5 Å) | 0.01 (Å)         |
|--modqp            |   | Use modified QASP from VirtualDrug or AD4 original    | 0 (no, use AD4)  |
|--cgmaps           |   | Use individual maps for CG-G0 instead of the same one | 0 (no, same map) |
|--stopstd          |   | AutoStop energy standard deviation tolerance          | 0.15 (kcal/mol)  |
|--initswgens       |   | Initial # generations of Solis-Wets instead of -lsmet | 0 (no)           |
|--gfpop            |   | Output all poses from all populations of each LGA run | 0 (no)           |
|--npdb             |   | # pose pdbqt files from populations of each LGA run   | 0                |
|--gbest            |   | Output single best pose as pdbqt file                 | 0 (no)           |

Autostop is ON by default since v1.4. The collective distribution of scores among all LGA populations
is tested for convergence every `<asfreq>` generations, and docking is stopped if the top-scored poses
exhibit a small variance. This avoids wasting computation after the best docking solutions have been found.
The heuristics set the number of evaluations at a generously large number. They are a function
of the number of rotatable bonds. It prevents unreasonably long dockings in cases where autostop fails
to detect convergence.
In our experience `--heuristics 1` and `--autostop 1` allow sufficient score evaluations for searching
the energy landscape accurately. For molecules with many rotatable bonds (e.g. about 15 or more)
it may be advisable to increase `--heurmax`.

When the heuristics is used and `--nev <max evals>` is provided as a command line argument it provides the (hard) upper # of evals limit to the value the heuristics suggests. Conversely, `--heurmax` is the rolling-off type asymptotic limit to the heuristic's # of evals formula and should only be changed with caution.
The batch file is a text file containing the parameters to `--ffile`, `--lfile`, and `--resnam` each on an individual line. It is possible to only use one line to specify the Protein grid map file which means it will be used for all ligands. Here is an example:
```
./receptor1.maps.fld
./ligand1.pdbqt
Ligand 1
./receptor2.maps.fld
./ligand2.pdbqt
Ligand 2
./receptor3.maps.fld
./ligand3.pdbqt
Ligand 3
```

When the distance-based analysis is used (`--contact_analysis 1` or `--contact_analysis <R_cutoff>,<H_cutoff>,<V_cutoff>`),
the ligand poses of a given run (either after a docking run or even when `--xml2dlg <xml file(s)>` is used) are analyzed in
terms of their individual atom distances to the target protein with individual cutoffs for:
* `R`eactive (default: 2.1 Å): These are interactions between modified atom types numbered 1, 4, or 7 (i.e. between C1 and S4)
* `H`ydrogen bonds (default: 3.7 Å): Interactions between Hydrogen-bond donor (closest N,O,S to an HD, or HD otherwise) and acceptor atom types (NA,NS,OA,OS,SA atom types).
* `V`an der Waals (default: 4.0  Å): All other interactions not fulfilling the above criteria.

The contact analysis results for each pose are output in dlg lines starting with `ANALYSIS:` and/or in `<contact_analysis>` blocks in xml file output.

# Documentation

Visit the project [Wiki](https://github.com/ccsb-scripps/AutoDock-GPU/wiki).
