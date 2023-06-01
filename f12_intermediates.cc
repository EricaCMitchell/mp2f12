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

#include "f12_intermediates.h"

#include "einsums/LinearAlgebra.hpp"
#include "einsums/Sort.hpp"
#include "einsums/Tensor.hpp"
#include "einsums/TensorAlgebra.hpp"

namespace psi{ namespace mp2f12 {

void f_mats(einsums::Tensor<double, 2> *f, einsums::Tensor<double, 2> *k, 
            einsums::Tensor<double, 2> *fk, einsums::Tensor<double, 2> *h,
            einsums::Tensor<double, 4> *G, const int& nocc, const int& nri) 
{
    using namespace einsums;
    using namespace tensor_algebra;
    using namespace tensor_algebra::index;

    (*f) = (*h)(All, All);
    {
        Tensor Id = create_identity_tensor("I", nocc, nocc);
        TensorView<double, 4> G1_view{(*G), Dim<4>{nri, nocc, nri, nocc}};
        TensorView<double, 4> G2_view{(*G), Dim<4>{nri, nocc, nocc, nri}};
        auto G1_pqiI = std::make_unique<Tensor<double, 4>>("pqiI", nri, nri, nocc, nocc);
        auto G2_pqiI = std::make_unique<Tensor<double, 4>>("pqiI", nri, nri, nocc, nocc);

        sort(Indices{p, q, i, I}, &G1_pqiI, Indices{p, i, q, I}, G1_view);
        sort(Indices{p, q, i, I}, &G2_pqiI, Indices{p, i, I, q}, G2_view);

        einsum(1.0, Indices{p, q}, &(*f), 2.0, Indices{p, q, i, I}, G1_pqiI, Indices{i, I}, Id);
        einsum(Indices{p, q}, &(*k), Indices{p, q, i, I}, G2_pqiI, Indices{i, I}, Id);
    }

    (*fk) = (*f)(All, All);
    tensor_algebra::element([](double const &val1, double const &val2)
                            -> double { return val1 - val2; },
                            &(*f), *k);
}

void V_and_X_mat(einsums::Tensor<double, 4> *VX, einsums::Tensor<double, 4> *F, einsums::Tensor<double, 4> *G_F, 
           einsums::Tensor<double, 4> *FG_F2, const int& nocc, const int& nobs, const int& nri)
{
    using namespace einsums;
    using namespace tensor_algebra;
    using namespace tensor_algebra::index;

    if ((*VX).name() == "V Intermediate") {
        (*VX) = (*FG_F2)(Range{0, nocc}, Range{0, nocc}, Range{0, nocc}, Range{0, nocc});
    } else {
        (*VX) = (*FG_F2)(Range{0, nocc}, Range{0, nocc}, Range{0, nocc}, Range{0, nocc});
    }

    Tensor F_ooco = (*F)(Range{0, nocc}, Range{0, nocc}, Range{nobs, nri}, Range{0, nocc});
    Tensor F_oooc = (*F)(Range{0, nocc}, Range{0, nocc}, Range{0, nocc}, Range{nobs, nri});
    Tensor F_oopq = (*F)(Range{0, nocc}, Range{0, nocc}, Range{0, nobs}, Range{0, nobs});
    Tensor G_F_ooco = (*G_F)(Range{0, nocc}, Range{0, nocc}, Range{nobs, nri}, Range{0, nocc});
    Tensor G_F_oooc = (*G_F)(Range{0, nocc}, Range{0, nocc}, Range{0, nocc}, Range{nobs, nri});
    Tensor G_F_oopq = (*G_F)(Range{0, nocc}, Range{0, nocc}, Range{0, nobs}, Range{0, nobs});

    einsum(1.0, Indices{i, j, k, l}, &(*VX), -1.0, Indices{i, j, p, n}, G_F_ooco, Indices{k, l, p, n}, F_ooco);
    einsum(1.0, Indices{i, j, k, l}, &(*VX), -1.0, Indices{i, j, m, q}, G_F_oooc, Indices{k, l, m, q}, F_oooc);
    einsum(1.0, Indices{i, j, k, l}, &(*VX), -1.0, Indices{i, j, p, q}, G_F_oopq, Indices{k, l, p, q}, F_oopq);
}

void C_mat(einsums::Tensor<double, 4> *C, einsums::Tensor<double, 4> *F, einsums::Tensor<double, 2> *f,
           const int& nocc, const int& nobs, const int& nri)
{
    using namespace einsums;
    using namespace tensor_algebra;
    using namespace tensor_algebra::index;

    auto nvir = nobs - nocc;
    Tensor F_oovc = (*F)(Range{0, nocc}, Range{0, nocc}, Range{nocc, nobs}, Range{nobs, nri});
    Tensor f_vc = (*f)(Range{nocc, nobs}, Range{nobs, nri});
    Tensor<double, 4> tmp{"Temp", nocc, nocc, nvir, nvir};

    einsum(Indices{k, l, a, b}, &tmp, Indices{k, l, a, q}, F_oovc, Indices{b, q}, f_vc);
    sort(Indices{k, l, a, b}, &(*C), Indices{k, l, a, b}, tmp);
    sort(1.0, Indices{k, l, a, b}, &(*C), 1.0, Indices{l, k, b, a}, tmp);

}

void B_mat(einsums::Tensor<double, 4> *B, einsums::Tensor<double, 4> *Uf, einsums::Tensor<double, 4> *F2,
           einsums::Tensor<double, 4> *F, einsums::Tensor<double, 2> *f, einsums::Tensor<double, 2> *fk, 
           einsums::Tensor<double, 2> *kk, const int& nocc, const int& nobs, const int& ncabs, const int& nri)
{
    using namespace einsums;
    using namespace tensor_algebra;
    using namespace tensor_algebra::index;

    auto nvir = nobs - nocc;
    Tensor<double, 4> B_nosymm{"B_klmn", nocc, nocc, nocc, nocc};

    B_nosymm = (*Uf)(Range{0, nocc}, Range{0, nocc}, Range{0, nocc}, Range{0, nocc});

    auto tmp_1 = std::make_unique<Tensor<double, 4>>("Temp 1", nocc, nocc, nocc, nocc);
    {
        Tensor F2_ooo1 = (*F2)(Range{0, nocc}, Range{0, nocc}, Range{0, nocc}, All);
        Tensor fk_o1   = (*fk)(Range{0, nocc}, All);

        einsum(Indices{l, k, n, m}, &tmp_1, Indices{l, k, n, I}, F2_ooo1, Indices{m, I}, fk_o1);
        sort(1.0, Indices{k, l, m, n}, &B_nosymm, 1.0, Indices{k, l, m, n}, *tmp_1);
        sort(1.0, Indices{k, l, m, n}, &B_nosymm, 1.0, Indices{l, k, n, m}, *tmp_1);
    }
    tmp_1.reset();

    auto tmp_2 = std::make_unique<Tensor<double, 4>>("Temp 2", nocc, nocc, nocc, nocc);
    {
        Tensor F_oo11 = (*F)(Range{0, nocc}, Range{0, nocc}, All, All);
        tmp_1 = std::make_unique<Tensor<double, 4>>("Temp 1", nocc, nocc, nri, nri);

        einsum(Indices{l, k, P, A}, &tmp_1, Indices{l, k, P, C}, F_oo11, Indices{C, A}, *kk);
        einsum(Indices{l, k, n, m}, &tmp_2, Indices{l, k, P, A}, tmp_1, Indices{n, m, P, A}, F_oo11);
        sort(1.0, Indices{k, l, m, n}, &B_nosymm, -1.0, Indices{k, l, m, n}, *tmp_2);
        sort(1.0, Indices{k, l, m, n}, &B_nosymm, -1.0, Indices{l, k, n, m}, *tmp_2);
    }
    tmp_1.reset();
    tmp_2.reset();

    {
        Tensor F_ooo1 = (*F)(Range{0, nocc}, Range{0, nocc}, Range{0, nocc}, All);
        tmp_1 = std::make_unique<Tensor<double, 4>>("Temp 1", nocc, nocc, nocc, nri);
        tmp_2 = std::make_unique<Tensor<double, 4>>("Temp 2", nocc, nocc, nocc, nocc);

        einsum(Indices{l, k, j, A}, &tmp_1, Indices{l, k, j, C}, F_ooo1, Indices{C, A}, *f);
        einsum(Indices{l, k, n, m}, &tmp_2, Indices{l, k, j, A}, tmp_1, Indices{n, m, j, A}, F_ooo1);
        sort(1.0, Indices{k, l, m, n}, &B_nosymm, -1.0, Indices{k, l, m, n}, *tmp_2);
        sort(1.0, Indices{k, l, m, n}, &B_nosymm, -1.0, Indices{l, k, n, m}, *tmp_2);
    }
    tmp_1.reset();
    tmp_2.reset();

    TensorView<double, 4> F_ooco_temp{(*F), Dim<4>{nocc, nocc, ncabs, nocc}, Offset<4>{0, 0, nobs, 0}};
    {
        Tensor F_ooco = F_ooco_temp;
        Tensor f_oo   = (*f)(Range{0, nocc}, Range{0, nocc});
        tmp_1 = std::make_unique<Tensor<double, 4>>("Temp 1", nocc, nocc, ncabs, nocc);
        tmp_2 = std::make_unique<Tensor<double, 4>>("Temp 2", nocc, nocc, nocc, nocc);

        einsum(Indices{l, k, p, i}, &tmp_1, Indices{l, k, p, j}, F_ooco, Indices{j, i}, f_oo);
        einsum(Indices{l, k, n, m}, &tmp_2, Indices{l, k, p, i}, tmp_1, Indices{n, m, p, i}, F_ooco);
        sort(1.0, Indices{k, l, m, n}, &B_nosymm, 1.0, Indices{k, l, m, n}, *tmp_2);
        sort(1.0, Indices{k, l, m, n}, &B_nosymm, 1.0, Indices{l, k, n, m}, *tmp_2);
    }
    tmp_1.reset();
    tmp_2.reset();

    TensorView<double, 4> F_oovq_temp{(*F), Dim<4>{nocc, nocc, nvir, nobs}, Offset<4>{0, 0, nocc, 0}};
    {
        Tensor F_oovq = F_oovq_temp;
        Tensor f_pq = (*f)(Range{0, nobs}, Range{0, nobs});
        tmp_1 = std::make_unique<Tensor<double, 4>>("Temp 1", nocc, nocc, nvir, nobs);
        tmp_2 = std::make_unique<Tensor<double, 4>>("Temp 2", nocc, nocc, nocc, nocc);

        einsum(Indices{l, k, b, p}, &tmp_1, Indices{l, k, b, r}, F_oovq, Indices{r, p}, f_pq);
        einsum(Indices{l, k, n, m}, &tmp_2, Indices{l, k, b, p}, tmp_1, Indices{n, m, b, p}, F_oovq);
        sort(1.0, Indices{k, l, m, n}, &B_nosymm, -1.0, Indices{k, l, m, n}, *tmp_2);
        sort(1.0, Indices{k, l, m, n}, &B_nosymm, -1.0, Indices{l, k, n, m}, *tmp_2);
    }
    tmp_1.reset();
    tmp_2.reset();

    {
        Tensor F_ooco = F_ooco_temp;
        Tensor F_ooc1 = (*F)(Range{0, nocc}, Range{0, nocc}, Range{nobs, nri}, All);
        Tensor f_o1   = (*f)(Range{0, nocc}, All);
        tmp_1 = std::make_unique<Tensor<double, 4>>("Temp 1", nocc, nocc, ncabs, nocc);
        tmp_2 = std::make_unique<Tensor<double, 4>>("Temp 2", nocc, nocc, nocc, nocc);

        einsum(Indices{l, k, p, j}, &tmp_1, Indices{l, k, p, I}, F_ooc1, Indices{j, I}, f_o1);
        einsum(0.0, Indices{l, k, n, m}, &tmp_2, 2.0, Indices{l, k, p, j}, tmp_1, Indices{n, m, p, j}, F_ooco);
        sort(1.0, Indices{k, l, m, n}, &B_nosymm, -1.0, Indices{k, l, m, n}, *tmp_2);
        sort(1.0, Indices{k, l, m, n}, &B_nosymm, -1.0, Indices{l, k, n, m}, *tmp_2);
    }
    tmp_1.reset();
    tmp_2.reset();

    {
        Tensor F_oovq = F_oovq_temp;
        Tensor F_oovc = (*F)(Range{0, nocc}, Range{0, nocc}, Range{nocc, nobs}, Range{nobs, nri});
        Tensor f_pc   = (*f)(Range{0, nobs}, Range{nobs, nri});
        tmp_1 = std::make_unique<Tensor<double, 4>>("Temp 1", nocc, nocc, nvir, ncabs);
        tmp_2 = std::make_unique<Tensor<double, 4>>("Temp 2", nocc, nocc, nocc, nocc);

        einsum(Indices{l, k, b, q}, &tmp_1, Indices{l, k, b, r}, F_oovq, Indices{r, q}, f_pc);
        einsum(0.0, Indices{l, k, n, m}, &tmp_2, 2.0, Indices{l, k, b, q}, tmp_1, Indices{n, m, b, q}, F_oovc);
        sort(1.0, Indices{k, l, m, n}, &B_nosymm, -1.0, Indices{k, l, m, n}, *tmp_2);
        sort(1.0, Indices{k, l, m, n}, &B_nosymm, -1.0, Indices{l, k, n, m}, *tmp_2);
    }
    tmp_1.reset();
    tmp_2.reset();

    (*B) = B_nosymm(All, All, All, All);
    sort(0.5, Indices{m, n, k, l}, &(*B), 0.5, Indices{k, l, m, n}, B_nosymm);
}

}} // end namespaces