---
name: Bug report
about: Create a bug report to help improve this project
title: ''
labels: ''
assignees: atillack

---

**Describe the bug**
A clear and concise description of what the bug is along with relevant output.

**To Reproduce**
Show us how to reproduce the failure. Please include which options are used (i.e. `--filelist`, `--import_dpf`). If you can, please also share with us the necessary files (receptor and ligand PDBQT files, AutoDock maps or DLG files, etc,..).

**Expected behavior**
A clear and concise description of what you expected to happen.

**Information to help narrow down the bug**
- Which version of AutoDock-GPU are you using?
- Which operating system are you on?
- Which compiler, compiler version, and `make` compile options did you use?
- Which GPU(s) are you running on and is Cuda or OpenCL used?
- Which driver version and if applicable, which Cuda version are you using?
- When compiling AutoDock-GPU, are `GPU_INCLUDE_PATH` and `GPU_LIBRARY_PATH` set? Are both environment variables set to the correct directories, i.e. corresponding to the correct Cuda version or OpenCL library?
- Did this bug only show up recently? Which version of AutoDock-GPU, compiler, settings, etc. were you using that worked?
