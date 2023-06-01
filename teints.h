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

#include "psi4/libmints/basisset.h"
#include "psi4/libmints/integralparameters.h"
#include "psi4/libmints/orbitalspace.h"
#include "psi4/libmints/typedefs.h"

#include "einsums/Tensor.hpp"

namespace psi{ namespace mp2f12 {

void convert_C(einsums::Tensor<double,2> *C, std::shared_ptr<BasisSet> bs);

void set_ERI(einsums::TensorView<double, 4>& ERI_Slice, einsums::Tensor<double, 4> *Slice);
void set_ERI(einsums::TensorView<double, 3>& ERI_Slice, einsums::Tensor<double, 3> *Slice);

void teints(const std::string& int_type, einsums::Tensor<double, 4> *ERI, std::vector<OrbitalSpace>& bs, 
            const int& nobs, std::shared_ptr<CorrelationFactor> corr);

void metric_ints(einsums::Tensor<double, 3> *DF_ERI, const std::vector<OrbitalSpace>& bs,
                 const std::shared_ptr<BasisSet> dfbs, const int& nobs, const int& naux, const int& nri);

void oper_ints(const std::string& int_type, einsums::Tensor<double, 3> *DF_ERI, einsums::Tensor<double, 2> *AB,
           std::vector<OrbitalSpace>& bs, std::shared_ptr<BasisSet> dfbs,
           const int& nobs, const int& naux, std::shared_ptr<CorrelationFactor> corr);

void df_teints(const std::string& int_type, einsums::Tensor<double, 4> *ERI, einsums::Tensor<double, 3> *Metric,
            std::vector<OrbitalSpace>& bs, std::shared_ptr<BasisSet> dfbs,
            const int& nobs, const int& naux, const int& nri, std::shared_ptr<CorrelationFactor> corr);

}} // end namespaces