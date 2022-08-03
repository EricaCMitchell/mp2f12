#
# @BEGIN LICENSE
#
# mp2f12 by Psi4 Developer, a plugin to:
#
# Psi4: an open-source quantum chemistry software package
#
# Copyright (c) 2007-2021 The Psi4 Developers.
#
# The copyrights for code used from other parties are included in
# the corresponding files.
#
# This file is part of Psi4.
#
# Psi4 is free software; you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, version 3.
#
# Psi4 is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License along
# with Psi4; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
#
# @END LICENSE
#

import psi4
import psi4.driver.p4util as p4util
from psi4.driver.procrouting import proc_util
from psi4.core import OrbitalSpace

def run_mp2f12(name, **kwargs):
    r"""Function encoding sequence of PSI module and plugin calls so that
    mp2f12 can be called via :py:func:`~driver.energy`. For post-scf plugins.

    >>> energy('mp2f12')

    """
    lowername = name.lower()
    kwargs = p4util.kwargs_lower(kwargs)

    # Your plugin's psi4 run sequence goes here
    psi4.core.set_local_option('MP2F12', 'PRINT', 0)

    # Compute a SCF reference, a wavefunction is return which holds the molecule used, orbitals
    # Fock matrices, and more
    print('Attention! This SCF may be density-fitted.')
    ref_wfn = kwargs.get('ref_wfn', None)
    if ref_wfn is None:
        #ref_wfn = psi4.driver.scf_helper(name, **kwargs)
        e_mp2, ref_wfn = psi4.driver.energy('mp2', return_wfn = True)
        cabs = build_cabs(ref_wfn)
    ref_wfn.set_basisset('CABS', cabs)

    # Ensure IWL files have been written when not using DF/CD
    proc_util.check_iwl_file_from_scf_type(psi4.core.get_option('SCF', 'SCF_TYPE'), ref_wfn)

    # Call the Psi4 plugin
    # Please note that setting the reference wavefunction in this way is ONLY for plugins
    mp2f12_wfn = psi4.core.plugin('mp2f12.so', ref_wfn)

    return mp2f12_wfn


# Integration with driver routines
psi4.driver.procedures['energy']['mp2f12'] = run_mp2f12

def build_cabs(wfn):
    '''
    Builds and returns CABS
    Provide wave function from RHF,
    OBS, and tolerance for linear dependence
    '''
    keys = ['BASIS','CABS_BASIS']
    targets = []
    roles = ['ORBITAL','F12']
    others = []
    targets.append(psi4.core.get_global_option('BASIS'))
    targets.append(psi4.core.get_global_option('CABS_BASIS'))
    others.append(psi4.core.get_global_option('BASIS'))
    others.append(psi4.core.get_global_option('BASIS'))

    # Creates combined basis set in Python
    mol = wfn.molecule()
    combined = psi4.driver.qcdb.libmintsbasisset.BasisSet.pyconstruct_combined(mol.save_string_xyz(),keys,targets,roles,others)
    cabs = psi4.core.BasisSet.construct_from_pydict(mol,combined,combined['puream'])
    return cabs

