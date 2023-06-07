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

#include "einsums/LinearAlgebra.hpp"
#include "einsums/Sort.hpp"
#include "einsums/Tensor.hpp"
#include "einsums/TensorAlgebra.hpp"

namespace psi { namespace mp2f12 {

void MP2F12::form_fock(einsums::Tensor<double, 2> *f, einsums::Tensor<double, 2> *k, 
                       einsums::Tensor<double, 2> *fk, einsums::Tensor<double, 2> *h,
                       einsums::Tensor<double, 4> *G)
{
    using namespace einsums;
    using namespace tensor_algebra;
    using namespace tensor_algebra::index;

    (*f) = (*h)(All, All);
    {
        Tensor Id = create_identity_tensor("I", nocc_, nocc_);
        TensorView<double, 4> G1_view{(*G), Dim<4>{nri_, nocc_, nri_, nocc_}};
        TensorView<double, 4> G2_view{(*G), Dim<4>{nri_, nocc_, nocc_, nri_}};
        auto G1_pqiI = std::make_unique<Tensor<double, 4>>("pqiI", nri_, nri_, nocc_, nocc_);
        auto G2_pqiI = std::make_unique<Tensor<double, 4>>("pqiI", nri_, nri_, nocc_, nocc_);

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

void MP2F12::form_V_or_X(einsums::Tensor<double, 4> *VX, einsums::Tensor<double, 4> *F,
                         einsums::Tensor<double, 4> *G_F, einsums::Tensor<double, 4> *FG_F2)
{
    using namespace einsums;
    using namespace tensor_algebra;
    using namespace tensor_algebra::index;

    (*VX) = (*FG_F2)(Range{0, nocc_}, Range{0, nocc_}, Range{0, nocc_}, Range{0, nocc_});

    {
        Tensor F_ooco = (*F)(Range{0, nocc_}, Range{0, nocc_}, Range{nobs_, nri_}, Range{0, nocc_});
        Tensor G_F_ooco = (*G_F)(Range{0, nocc_}, Range{0, nocc_}, Range{nobs_, nri_}, Range{0, nocc_});
        einsum(1.0, Indices{i, j, k, l}, &(*VX), -1.0, Indices{i, j, p, n}, G_F_ooco, Indices{k, l, p, n}, F_ooco);
    }

    {
        Tensor F_oooc = (*F)(Range{0, nocc_}, Range{0, nocc_}, Range{0, nocc_}, Range{nobs_, nri_});
        Tensor G_F_oooc = (*G_F)(Range{0, nocc_}, Range{0, nocc_}, Range{0, nocc_}, Range{nobs_, nri_});
        einsum(1.0, Indices{i, j, k, l}, &(*VX), -1.0, Indices{i, j, m, q}, G_F_oooc, Indices{k, l, m, q}, F_oooc);
    }

    {
        Tensor F_oopq = (*F)(Range{0, nocc_}, Range{0, nocc_}, Range{0, nobs_}, Range{0, nobs_});
        Tensor G_F_oopq = (*G_F)(Range{0, nocc_}, Range{0, nocc_}, Range{0, nobs_}, Range{0, nobs_});
        einsum(1.0, Indices{i, j, k, l}, &(*VX), -1.0, Indices{i, j, p, q}, G_F_oopq, Indices{k, l, p, q}, F_oopq);
    }
}

void MP2F12::form_C(einsums::Tensor<double, 4> *C, einsums::Tensor<double, 4> *F,
                    einsums::Tensor<double, 2> *f)
{
    using namespace einsums;
    using namespace tensor_algebra;
    using namespace tensor_algebra::index;

    Tensor F_oovc = (*F)(Range{0, nocc_}, Range{0, nocc_}, Range{nocc_, nobs_}, Range{nobs_, nri_});
    Tensor f_vc = (*f)(Range{nocc_, nobs_}, Range{nobs_, nri_});
    Tensor<double, 4> tmp{"Temp", nocc_, nocc_, nvir_, nvir_};

    einsum(Indices{k, l, a, b}, &tmp, Indices{k, l, a, q}, F_oovc, Indices{b, q}, f_vc);
    sort(Indices{k, l, a, b}, &(*C), Indices{k, l, a, b}, tmp);
    sort(1.0, Indices{k, l, a, b}, &(*C), 1.0, Indices{l, k, b, a}, tmp);

}

void MP2F12::form_B(einsums::Tensor<double, 4> *B, einsums::Tensor<double, 4> *Uf,
                    einsums::Tensor<double, 4> *F2, einsums::Tensor<double, 4> *F,
                    einsums::Tensor<double, 2> *f, einsums::Tensor<double, 2> *fk,
                    einsums::Tensor<double, 2> *kk)
{
    using namespace einsums;
    using namespace tensor_algebra;
    using namespace tensor_algebra::index;

    Tensor<double, 4> B_nosymm{"B_klmn", nocc_, nocc_, nocc_, nocc_};

    B_nosymm = (*Uf)(Range{0, nocc_}, Range{0, nocc_}, Range{0, nocc_}, Range{0, nocc_});

    auto tmp_1 = std::make_unique<Tensor<double, 4>>("Temp 1", nocc_, nocc_, nocc_, nocc_);
    {
        Tensor F2_ooo1 = (*F2)(Range{0, nocc_}, Range{0, nocc_}, Range{0, nocc_}, All);
        Tensor fk_o1   = (*fk)(Range{0, nocc_}, All);

        einsum(Indices{l, k, n, m}, &tmp_1, Indices{l, k, n, I}, F2_ooo1, Indices{m, I}, fk_o1);
        sort(1.0, Indices{k, l, m, n}, &B_nosymm, 1.0, Indices{k, l, m, n}, *tmp_1);
        sort(1.0, Indices{k, l, m, n}, &B_nosymm, 1.0, Indices{l, k, n, m}, *tmp_1);
    }
    tmp_1.reset();

    auto tmp_2 = std::make_unique<Tensor<double, 4>>("Temp 2", nocc_, nocc_, nocc_, nocc_);
    {
        Tensor F_oo11 = (*F)(Range{0, nocc_}, Range{0, nocc_}, All, All);
        tmp_1 = std::make_unique<Tensor<double, 4>>("Temp 1", nocc_, nocc_, nri_, nri_);

        einsum(Indices{l, k, P, A}, &tmp_1, Indices{l, k, P, C}, F_oo11, Indices{C, A}, *kk);
        einsum(Indices{l, k, n, m}, &tmp_2, Indices{l, k, P, A}, tmp_1, Indices{n, m, P, A}, F_oo11);
        sort(1.0, Indices{k, l, m, n}, &B_nosymm, -1.0, Indices{k, l, m, n}, *tmp_2);
        sort(1.0, Indices{k, l, m, n}, &B_nosymm, -1.0, Indices{l, k, n, m}, *tmp_2);
    }
    tmp_1.reset();
    tmp_2.reset();

    {
        Tensor F_ooo1 = (*F)(Range{0, nocc_}, Range{0, nocc_}, Range{0, nocc_}, All);
        tmp_1 = std::make_unique<Tensor<double, 4>>("Temp 1", nocc_, nocc_, nocc_, nri_);
        tmp_2 = std::make_unique<Tensor<double, 4>>("Temp 2", nocc_, nocc_, nocc_, nocc_);

        einsum(Indices{l, k, j, A}, &tmp_1, Indices{l, k, j, C}, F_ooo1, Indices{C, A}, *f);
        einsum(Indices{l, k, n, m}, &tmp_2, Indices{l, k, j, A}, tmp_1, Indices{n, m, j, A}, F_ooo1);
        sort(1.0, Indices{k, l, m, n}, &B_nosymm, -1.0, Indices{k, l, m, n}, *tmp_2);
        sort(1.0, Indices{k, l, m, n}, &B_nosymm, -1.0, Indices{l, k, n, m}, *tmp_2);
    }
    tmp_1.reset();
    tmp_2.reset();

    TensorView<double, 4> F_ooco_temp{(*F), Dim<4>{nocc_, nocc_, ncabs_, nocc_}, Offset<4>{0, 0, nobs_, 0}};
    {
        Tensor F_ooco = F_ooco_temp;
        Tensor f_oo   = (*f)(Range{0, nocc_}, Range{0, nocc_});
        tmp_1 = std::make_unique<Tensor<double, 4>>("Temp 1", nocc_, nocc_, ncabs_, nocc_);
        tmp_2 = std::make_unique<Tensor<double, 4>>("Temp 2", nocc_, nocc_, nocc_, nocc_);

        einsum(Indices{l, k, p, i}, &tmp_1, Indices{l, k, p, j}, F_ooco, Indices{j, i}, f_oo);
        einsum(Indices{l, k, n, m}, &tmp_2, Indices{l, k, p, i}, tmp_1, Indices{n, m, p, i}, F_ooco);
        sort(1.0, Indices{k, l, m, n}, &B_nosymm, 1.0, Indices{k, l, m, n}, *tmp_2);
        sort(1.0, Indices{k, l, m, n}, &B_nosymm, 1.0, Indices{l, k, n, m}, *tmp_2);
    }
    tmp_1.reset();
    tmp_2.reset();

    TensorView<double, 4> F_oovq_temp{(*F), Dim<4>{nocc_, nocc_, nvir_, nobs_}, Offset<4>{0, 0, nocc_, 0}};
    {
        Tensor F_oovq = F_oovq_temp;
        Tensor f_pq = (*f)(Range{0, nobs_}, Range{0, nobs_});
        tmp_1 = std::make_unique<Tensor<double, 4>>("Temp 1", nocc_, nocc_, nvir_, nobs_);
        tmp_2 = std::make_unique<Tensor<double, 4>>("Temp 2", nocc_, nocc_, nocc_, nocc_);

        einsum(Indices{l, k, b, p}, &tmp_1, Indices{l, k, b, r}, F_oovq, Indices{r, p}, f_pq);
        einsum(Indices{l, k, n, m}, &tmp_2, Indices{l, k, b, p}, tmp_1, Indices{n, m, b, p}, F_oovq);
        sort(1.0, Indices{k, l, m, n}, &B_nosymm, -1.0, Indices{k, l, m, n}, *tmp_2);
        sort(1.0, Indices{k, l, m, n}, &B_nosymm, -1.0, Indices{l, k, n, m}, *tmp_2);
    }
    tmp_1.reset();
    tmp_2.reset();

    {
        Tensor F_ooco = F_ooco_temp;
        Tensor F_ooc1 = (*F)(Range{0, nocc_}, Range{0, nocc_}, Range{nobs_, nri_}, All);
        Tensor f_o1   = (*f)(Range{0, nocc_}, All);
        tmp_1 = std::make_unique<Tensor<double, 4>>("Temp 1", nocc_, nocc_, ncabs_, nocc_);
        tmp_2 = std::make_unique<Tensor<double, 4>>("Temp 2", nocc_, nocc_, nocc_, nocc_);

        einsum(Indices{l, k, p, j}, &tmp_1, Indices{l, k, p, I}, F_ooc1, Indices{j, I}, f_o1);
        einsum(0.0, Indices{l, k, n, m}, &tmp_2, 2.0, Indices{l, k, p, j}, tmp_1, Indices{n, m, p, j}, F_ooco);
        sort(1.0, Indices{k, l, m, n}, &B_nosymm, -1.0, Indices{k, l, m, n}, *tmp_2);
        sort(1.0, Indices{k, l, m, n}, &B_nosymm, -1.0, Indices{l, k, n, m}, *tmp_2);
    }
    tmp_1.reset();
    tmp_2.reset();

    {
        Tensor F_oovq = F_oovq_temp;
        Tensor F_oovc = (*F)(Range{0, nocc_}, Range{0, nocc_}, Range{nocc_, nobs_}, Range{nobs_, nri_});
        Tensor f_pc   = (*f)(Range{0, nobs_}, Range{nobs_, nri_});
        tmp_1 = std::make_unique<Tensor<double, 4>>("Temp 1", nocc_, nocc_, nvir_, ncabs_);
        tmp_2 = std::make_unique<Tensor<double, 4>>("Temp 2", nocc_, nocc_, nocc_, nocc_);

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