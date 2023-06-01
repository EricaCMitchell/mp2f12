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

#include "einsums/Tensor.hpp"

namespace psi{ namespace mp2f12 {

void f_mats(einsums::Tensor<double, 2> *f, einsums::Tensor<double, 2> *k, 
            einsums::Tensor<double, 2> *fk, einsums::Tensor<double, 2> *h,
            einsums::Tensor<double, 4> *G, const int& nocc, const int& nri);

void V_and_X_mat(einsums::Tensor<double, 4> *VX, einsums::Tensor<double, 4> *F, einsums::Tensor<double, 4> *G_F, 
           einsums::Tensor<double, 4> *FG_F2, const int& nocc, const int& nobs, const int& nri);

void C_mat(einsums::Tensor<double, 4> *C, einsums::Tensor<double, 4> *F, einsums::Tensor<double, 2> *f,
           const int& nocc, const int& nobs, const int& nri);

void B_mat(einsums::Tensor<double, 4> *B, einsums::Tensor<double, 4> *Uf, einsums::Tensor<double, 4> *F2,
           einsums::Tensor<double, 4> *F, einsums::Tensor<double, 2> *f, einsums::Tensor<double, 2> *fk, 
           einsums::Tensor<double, 2> *kk, const int& nocc, const int& nobs, const int& ncabs, const int& nri);

}} // end namespaces