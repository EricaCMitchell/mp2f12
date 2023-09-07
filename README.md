# mp2f12
An explicity correlated second-order Moeller Plesset perturbation theory plugin to Psi4.

# Overview
This plugin to Psi4 performs explicity correlated second-order Moeller Plesset perturbation theory (MP2-F12) computations using the most robust ansatz (3C) within the fixed or SP- ansatz of Ten-no (FIX) most commonly seen as MP2-F12/3C(FIX). This particular version of the method scales _N_<sup>5</sup>. The bulk of the time to compute this plugin is dedicated to formation of the two-electron integrals, which are not screened due to limitations with Psi4.

# Installation
Only two main depedencies are needed, with the rest being needed by Psi4 as well.

## Psi4
Psi4 may be installed with a conda environment or from source.

### Anaconda Envrionment
A new version of python may also be used just check the [Psi4 website](https://psicode.org/).
```
conda install psi4 python=3.10 -c conda-forge/label/libint_dev -c conda-forge
conda install cmake
```

### Source
Two steps will be needed as of 7 Sep 2023.

1. Clone the Psi4 repo
```
git clone https://github.com/psi4/psi4.git
```

2. Pull the "CABS, rework build_cabs_space" PR
```
git fetch origin pull/2982/head:BRANCH_NAME
git checkout BRANCH_NAME
```

## EinsumsInCpp
EinsumsInCpp is a submodule of this project.

```
git submodule init
git submodule update
```

## mp2f12
Once the dependencies are loaded, the plugin can be made.
```
psi4 --plugin-compile -DEINSUMS_ENABLE_TESTING=ON
make
```

# Input Options
* **F12_TYPE** (string):
    
    Defines which algorithm to use. 
    Allowed values are CONV, DF, DISK_CONV, and DISK_DF. 
    The default value is CONV.

* **F12_BETA** (double):
    
    Slater exponent for contracted Gaussian-type geminal.
    Note that the exponents 0.9, 1.0, and 1.0 <MATH>_a_<sub>0</sub><sup>-1</sup></MATH>
    are recommended for cc-pVXZ-F12 where X = D, T, and Q, respectively, while those for
    aug-cc-pVXZ where X = D, T, and Q are 1.1, 1.2, and 1.4 <MATH>_a_<sub>0</sub><sup>-1</sup></MATH>.
    The default value is <MATH>_a_<sub>0</sub><sup>-1</sup></MATH>.

* **CABS_BASIS** (string):

    The CABS recommendations are
    - cc-pVXZ-F12 with cc-pVXZ-F12-OPTRI
    - aug-cc-pVXZ with cc-pVXZ-JKFIT

* **CABS_SINGLES** (bool):

    Turns on the singles correction, which accounts for
    the incompleteness error in the Hartree-Fock energy
    due to the introduction of the CABS. 
    Default value is TRUE.

* **DF_BASIS_F12** (string):

    Defines which basis set to use for density-fitting
    in the F12 computation. Also, sets the DF_BASIS_MP2
    for the MP2 computation.


