# BOPDMD.jl
Scripts to call the BOp-DMD algorithm in Julia.

For now, this is a standalone script, but I am hoping to integrate this code within the (DataDrivenDiffEq.jl)[https://docs.sciml.ai/DataDrivenDiffEq/stable/] and broader (SciML.jl) ecosystems, but I am a PhD student so this is not a top priority currently :) 

My development environment is on a NixOS-based machine, so an included `scientific-fhs` flake is included.

Python files are copied from the `PyDMD` repository and were the files I used to port the algorithm. The provided `test_bopdmd.py` and `test_bopdmd.jl` files show example usage of both versions of the code.

TODO:
    - Type annotations
    - Better parallelization of bags
    - Add `PyDMD` functionality like eigenvalue and mode plotting