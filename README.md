# MLLGfirstorder
First order FEM BEM coupling for the MLLG system with FENICS and BEMPP

Further details can be found in my thesis https://publikationen.bibliothek.kit.edu/1000133728 or https://doi.org/10.5445/IR/1000133728

run errorplottime.py or errorplotspace.py for a experimental order of convergence experiment either in time or in space.  
The computations are done in the MLLGfunc... files. Inputs are discretization parameters and material parameters, output is the (coefficients of the) solution on the corresponding time points.  Different versions are implemented optimised for either time stepping or space discretization. 

The software runs with Python3 and
Fenics 2019.1.0 
Bempp-cl 0.2.0
