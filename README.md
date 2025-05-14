## Note ##
This script is now contained within a larger collection of DMD scripts in Julia [here](https://github.com/michaelhess17/Julia_DMD_Multiverse). 

## What is this repository?
Scripts to call the Bagged, optimized DMD (BOp-DMD) algorithm in Julia using variable projection.

For now, this is a standalone script, but I am hoping to integrate this code within the [DataDrivenDiffEq.jl](https://docs.sciml.ai/DataDrivenDiffEq/stable/) and broader [SciML.jl](https://sciml.ai/) ecosystems, but I am a PhD student so this is not my only priority currently :) 

My development environment is on a NixOS-based machine, so I have included the `scientific-fhs` flake I use to develop. I also use `direnv` to automatically manage the development shell upon entering and exiting the project root folder. Settings provided in `.envrc` allow this to happen automatically after running `direnv allow`. 

Python files are copied from the `PyDMD` repository and were the files I used to port the algorithm. The provided `test_bopdmd.py` and `test_bopdmd.jl` files show example usage of both versions of the code.
