This is the documentation for developers: source code structure, build instructions, tools required, etc.

# Source-code structure
.
+-- hola
+-- amigos
|   +-- son todos
|   +-- muy
|   +-- buenos
|-- amigos

# Compilation

## Basic
```zsh
make DEVICE=<TYPE> NUMWI=<NWI>
```
`<TYPE>` : CPU, GPU.

`<NUMWI>` : 16, 32, 64

After successful compilation, the host binary `ocladock_<type>_<N>wi` is placed under [bin](./bin).

`type` denotes the accelerator chosen: `cpu` or `gpu`.

`N` denotes the OpenCL work-group size: `16`, `32`, or `64`.

## All available options
```zsh
make DEVICE=<TYPE> NUMWI=<NWI> CONFIG=<CFG> DOCK_DEBUG=<Y/N> REPRO=<Y/N>
```
| Argument    | Description                           | Possible values          |
|-------------|---------------------------------------|--------------------------|
| DEVICE      | OpenCL device type                    | `<TYPE>`: CPU, GPU       |
| NUMWI       | Number of work items per work group   | `<NWI>` : 16, 32, 64     |
| CONFIG      | Host configuration                    | `<CFG>` : DEBUG, RELEASE |
| DOCK_DEBUG  | Enable debug info from host & device  | `<Y/N>` : YES, NO        |
| REPRO       | Reproduce results (remove randomness) | `<Y/N>` : YES, NO        |

## Configuration file
Check the configurations in the project's [Makefile](../Makefile).

# Requirements
