/*
 * @BEGIN LICENSE
 *
 * MP2F12 by Erica Mitchell, a plugin to:
 *
 * Psi4: an open-source quantum chemistry software package
 *
 * Copyright (c) 2007-2023 The Psi4 Developers.
 *
 * The copyrights for code used from other parties are included in
 * the corresponding files.
 *
 * This file is part of Psi4.
 *
 * Psi4 is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * Psi4 is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License along
 * with Psi4; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 * @END LICENSE
 */

#include "mp2f12.h"

#include <psi4/libplugin/plugin.h>
#include <psi4/liboptions/liboptions.h>

namespace psi { namespace mp2f12 {

extern "C" PSI_API
int read_options(std::string name, Options& options)
{
    if (name == "MP2F12"|| options.read_globals()) {
        /*- The amount of information printed to the output file -*/
        options.add_int("PRINT", 0);
        /*- Choose a basis for Complementary Auxiliary Basis Set -*/
        options.add_str("CABS_BASIS", "");
        /*- Whether to compute the CABS Singles Correction -*/
        options.add_bool("CABS_SINGLES", true);
        /*- Choose conventional or density-fitted. Default to CONV -*/
        options.add_str("F12_TYPE", "CONV");
        /*- Choose a density-fitting basis for integrals -*/
        options.add_str("DF_BASIS_F12", "");
        /*- Set contracted Gaussian-type geminal beta value -*/
        options.add_double("F12_BETA", 1.0);
    }

    return true;
}

extern "C" PSI_API
SharedWavefunction mp2f12(SharedWavefunction ref_wfn, Options& options)
{
    
    std::shared_ptr<MP2F12> mp2f12(new MP2F12(ref_wfn, options));

    double e_f12_total = mp2f12->compute_energy();

    ref_wfn->set_scalar_variable("CURRENT ENERGY", e_f12_total);

    return ref_wfn;
}

}} // End namespaces
