import sys
sys.path.insert(0, '../..')
import psi4
import mp2f12
import pytest

@pytest.mark.parametrize("types,ref,tol", [("conv", -76.3582380027535, 9),
                                       ("df", -76.359179001859985, 6)])

def test_f12_type(types,ref,tol):

  h2o = psi4.geometry(
    """
    O    0.000000000000    -0.000000000044    -0.124307973994
    H    0.000000000000    -1.430467819606     0.986428877419
    H    0.000000000000     1.430467820301     0.986428877632

    units bohr
    symmetry c1
    """
  )

  psi4.set_options({"basis" : "cc-pvdz-f12",
                    "freeze_core" : True,
                    "e_convergence" : 1.e-10})

  psi4.set_module_options("mp2f12",
                          {"cabs_basis" : "cc-pvdz-f12-optri",
                           "df_basis_f12" : "aug-cc-pvdz-ri",
                           "f12_type" : types,
                           "cabs_singles" : True})

  _, wfn = psi4.energy('mp2f12', return_wfn=True)

  assert psi4.compare_values(wfn.variable("CURRENT ENERGY"), ref, tol, "Psi4 v. MPQC")
