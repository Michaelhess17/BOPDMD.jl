# BOPDMD.jl
Scripts to call the BOp-DMD algorithm in Julia.

For now, this is a standalone script, but I am hoping to integrate this code within the [DataDrivenDiffEq.jl](https://docs.sciml.ai/DataDrivenDiffEq/stable/) and broader [SciML.jl](https://sciml.ai/) ecosystems, but I am a PhD student so this is not a top priority currently :) 

My development environment is on a NixOS-based machine, so I have included the `scientific-fhs` flake I use to develop. I also use `direnv` to automatically manage the development shell upon entering and exiting the project root folder. Settings provided in `.envrc` allow this to happen automatically after running `direnv allow`. 

Python files are copied from the `PyDMD` repository and were the files I used to port the algorithm. The provided `test_bopdmd.py` and `test_bopdmd.jl` files show example usage of both versions of the code.

TODO:
- Type annotations
- Better parallelization of bags
- Add `PyDMD` functionality like eigenvalue and mode plotting
