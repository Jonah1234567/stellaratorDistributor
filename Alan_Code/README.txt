The files in this folder are as follows:
- input.test1
  An input file containing boundary fourier coefficients. 
  
- agTargets_34.py
  Contains various functions that are used to calculate the total penalty.

- Helpers3.py
  Contains several helper functions that are used in agTargets_34.py

- main_1.py
  This code is the simplest version of a possible initial condition finder.
  It simply takes an input file (input.test1) and calculates the target function penalty.

- main_2.py
  A more complicated initial condition finder.
  Takes an input file (input.test1) and runs an optimization to minimize the target function.

The unusual packages required to run these scripts are:
- booz_xform (pip install booz_xform)
- VMEC2000 (https://github.com/hiddenSymmetries/VMEC2000)
- SIMSOPT (https://github.com/hiddenSymmetries/simsopt)

Instructions for installing VMEC and SIMSOPT on various architectures can be found here:
https://github.com/hiddenSymmetries/simsopt/wiki

If using main_1, we'd want to vary the coefficients in input.test1, ideally with step size around 1e-6 or 1e-7.
The coefficients could have values between -1e-1 and +1e-1, for a total of around 2e5 or 2e6 runs.
Not all fourier coefficients would need to be varied, but ideally between 10 and 28 would be. 
This means that, ideally we'd want to do between 1e6 and 1e8 total executions.
The useful information from this would be:
  1 - The values in the 'input.test1' file
  2 - The 'wout_test1_000_000000.nc' file generated
  3 - The penalty function output

If using main_2, we'd want to vary the coefficients in input.test, although with much larger step size.
We would also want to test several input files, but fewer than we'd need in main_1, due to the optimization.
Ideally, we would test 2 or 3 abs_step values, and 2 diff_method values, for a total of 4 or 6.
The more modes optimized, the better, but ideally we'd want to optimize between 10 and 28 modes.
Two versions of this optimization can be tested by setting the variable 'smaller_opt' as 'True' or 'False'.
The useful information from this would be:
  1 - The values in the last 'input.test1_000_XXXXXX' file generated
  2 - The values in the last 'wout_test1_000_XXXXXX.nc' netcdf file generated
  3 - The values in the 'objective' file generated
