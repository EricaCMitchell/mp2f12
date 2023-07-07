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

#ifdef _OPENMP
#include <omp.h>
#include <psi4/libpsi4util/process.h>
#endif

#include "mp2f12.h"

#include <psi4/libmints/basisset.h>
#include <psi4/libmints/integral.h>
#include <psi4/libmints/matrix.h>
#include <psi4/libmints/mintshelper.h>
#include <psi4/libmints/onebody.h>
#include <psi4/libmints/orbitalspace.h>
#include <psi4/lib3index/dftensor.h>

#include "einsums/TensorAlgebra.hpp"
#include "einsums/Sort.hpp"
#include "einsums/Tensor.hpp"
#include "einsums/Print.hpp"

namespace psi { namespace mp2f12 {

void MP2F12::convert_C(einsums::Tensor<double,2> *C, OrbitalSpace bs, const int& dim1, const int& dim2)
{
#pragma omp parallel for collapse(2) num_threads(nthreads_)
    for (int p = 0; p < dim1; p++) {
        for (int q = 0; q < dim2; q++) {
            (*C)(p, q) = bs.C()->get(p, q);
        }
    }
}

void MP2F12::set_ERI(einsums::TensorView<double, 4>& ERI_Slice, einsums::Tensor<double, 4> *Slice)
{
    auto dim1 = (*Slice).dim(0);
    auto dim2 = (*Slice).dim(1);
    auto dim3 = (*Slice).dim(2);
    auto dim4 = (*Slice).dim(3);

#pragma omp parallel for collapse(4) num_threads(nthreads_)
    for (int p = 0; p < dim1; p++){
        for (int q = 0; q < dim2; q++){
            for (int r = 0; r < dim3; r++){
                for (int s = 0; s < dim4; s++){
                    ERI_Slice(p, q, r, s) = (*Slice)(p, q, r, s);
                }
            }
        }
    }
}

void MP2F12::set_ERI(einsums::TensorView<double, 3>& ERI_Slice, einsums::Tensor<double, 3> *Slice)
{
    auto naux_= (*Slice).dim(0);
    auto dim1 = (*Slice).dim(1);
    auto dim2 = (*Slice).dim(2);

#pragma omp parallel for collapse(3) num_threads(nthreads_)
    for (int A = 0; A < naux_; A++){
        for (int p = 0; p < dim1; p++){
            for (int q = 0; q < dim2; q++){
                ERI_Slice(A, p, q) = (*Slice)(A, p, q);
            }
        }
    }
}

void MP2F12::form_oeints(einsums::Tensor<double, 2> *h)
{ 
    using namespace einsums;

    outfile->Printf("   One-Electron Integrals\n");

    std::vector<int> order = {0, 0, 1, 
                              0, 1, 1};

    int nmo1, nmo2, M, N;
    for (int i = 0; i < 3; i++) {
        (  order[i] == 1  ) ? (nmo1 = ncabs_, M = nobs_) : (nmo1 = nobs_, M = 0);
        ( order[i+3] == 1 ) ? (nmo2 = ncabs_, N = nobs_) : (nmo2 = nobs_, N = 0);

        auto mints = reference_wavefunction_->mintshelper();

        // Transform OEI AO Matrix into OEI MO Matrix
        auto t_mo = std::make_shared<Matrix>("MO-based T Integral", nmo1, nmo2);
        auto v_mo = std::make_shared<Matrix>("MO-based V Integral", nmo1, nmo2);
        {
            auto bs1 = bs_[ order[i] ].basisset();
            auto bs2 = bs_[order[i+3]].basisset();
            auto C1 = bs_[ order[i] ].C();
            auto C2 = bs_[order[i+3]].C();
            auto t_ao = mints->ao_kinetic(bs1, bs2);
            auto v_ao = mints->ao_potential(bs1, bs2);
            t_mo->transform(C1, t_ao, C2);
            v_mo->transform(C1, v_ao, C2);
            t_ao.reset();
            v_ao.reset();
        }

        // Stitch into OEI
        {
            TensorView<double, 2> h_mn{*h, Dim<2>{nmo1, nmo2}, Offset<2>{M, N}};
#pragma omp parallel for collapse(2) num_threads(nthreads_)
            for (int m = 0; m < nmo1; m++) {
                for (int n = 0; n < nmo2; n++) {
                    h_mn(m, n) = t_mo->get(m, n) + v_mo->get(m, n);
                }
            }

            if ( order[i] != order[i+3] ) {
                TensorView<double, 2> h_nm{*h, Dim<2>{nmo2, nmo1}, Offset<2>{N, M}};
#pragma omp parallel for collapse(2) num_threads(nthreads_)
                for (int n = 0; n < nmo2; n++) {
                    for (int m = 0; m < nmo1; m++) {
                        h_nm(n, m) = t_mo->get(m, n) + v_mo->get(m, n);
                    }
                }
            } // end of if statement
        }
    } // end for loop
}

void MP2F12::form_teints(const std::string& int_type, einsums::Tensor<double, 4> *ERI)
{
    using namespace einsums;
    using namespace tensor_algebra;
    using namespace tensor_algebra::index;

    // In <PQ|RS> ordering
    std::vector<int> order = {'o', 'o', 'o', 'o'};
    if ( int_type == "F" ) {
        order = {'o', 'o', 'O', 'O',
                 'o', 'o', 'O', 'C',
                 'o', 'o', 'C', 'C'};
    } else if ( int_type == "F2" ) {
        order = {'o', 'o', 'o', 'O',
                 'o', 'o', 'o', 'C'};
    } else if ( int_type == "G" ) {
        order = {'o', 'o', 'O', 'O',
                 'o', 'o', 'O', 'C'};
    } else if ( int_type == "J" ) {
        order = {'O', 'o', 'O', 'o',
                 'O', 'o', 'C', 'o',
                 'C', 'o', 'C', 'o'};
    } else if ( int_type == "K" ) {
        order = {'O', 'o', 'o', 'O',
                 'O', 'o', 'o', 'C',
                 'C', 'o', 'o', 'C'};
    }

    int nmo1, nmo2, nmo3, nmo4;
    int I, J, K, L;
    int o1, o2, o3, o4;
    for (int idx = 0; idx < (order.size()/4); idx++) {
        int i = idx * 4;
        ( order[i]  == 'C') ? o1 = 1 : o1 = 0;
        (order[i+1] == 'C') ? o2 = 1 : o2 = 0;
        (order[i+2] == 'C') ? o3 = 1 : o3 = 0;
        (order[i+3] == 'C') ? o4 = 1 : o4 = 0;
        auto bs1 = bs_[o1].basisset();
        auto bs2 = bs_[o2].basisset();
        auto bs3 = bs_[o3].basisset();
        auto bs4 = bs_[o4].basisset();
        auto nbf1 = bs1->nbf();
        auto nbf2 = bs2->nbf();
        auto nbf3 = bs3->nbf();
        auto nbf4 = bs4->nbf();

        // Create ERI AO Tensor
        auto GAO = std::make_unique<Tensor<double, 4>>("ERI AO", nbf1, nbf2, nbf3, nbf4);
        {
            std::shared_ptr<IntegralFactory> intf(new IntegralFactory(bs1, bs3, bs2, bs4));

            std::vector<std::shared_ptr<TwoBodyAOInt>> ints;
            if ( int_type == "F" ){
                ints.push_back(std::shared_ptr<TwoBodyAOInt>(intf->f12(cgtg_)));
            } else if ( int_type == "FG" ){
                ints.push_back(std::shared_ptr<TwoBodyAOInt>(intf->f12g12(cgtg_)));
            } else if ( int_type == "F2" ){
                ints.push_back(std::shared_ptr<TwoBodyAOInt>(intf->f12_squared(cgtg_)));
            } else if ( int_type == "Uf" ){
                ints.push_back(std::shared_ptr<TwoBodyAOInt>(intf->f12_double_commutator(cgtg_)));
            } else { 
                ints.push_back(std::shared_ptr<TwoBodyAOInt>(intf->eri()));
            }
            
            // Make ints vector
            for (size_t thread = 1; thread < nthreads_; thread++) {
                ints.push_back(std::shared_ptr<TwoBodyAOInt>(ints[0]->clone()));
            }
            
#pragma omp parallel for collapse(4) schedule(guided) num_threads(nthreads_)
            for (size_t M = 0; M < bs1->nshell(); M++) {
                for (size_t N = 0; N < bs3->nshell(); N++) {
                    for (size_t P = 0; P < bs2->nshell(); P++) {
                        for (size_t Q = 0; Q < bs4->nshell(); Q++) {
                            size_t rank = 0;
#ifdef _OPENMP
                            rank = omp_get_thread_num();
#endif
                            const size_t numM = bs1->shell(M).nfunction();
                            const size_t numN = bs3->shell(N).nfunction();
                            const size_t numP = bs2->shell(P).nfunction();
                            const size_t numQ = bs4->shell(Q).nfunction();
                            const size_t index_M = bs1->shell(M).function_index();
                            const size_t index_N = bs3->shell(N).function_index();
                            const size_t index_P = bs2->shell(P).function_index();
                            const size_t index_Q = bs4->shell(Q).function_index();

                            ints[rank]->compute_shell(M, N, P, Q);
                            const auto *ints_buff = ints[rank]->buffers()[0];

                            size_t index = 0;
                            for (size_t m = 0; m < numM; m++) {
                                for (size_t n = 0; n < numN; n++) {
                                    for (size_t p = 0; p < numP; p++) {
                                        for (size_t q = 0; q < numQ; q++) {
                                            (*GAO)(index_M + m,
                                                   index_P + p,
                                                   index_N + n,
                                                   index_Q + q) = ints_buff[index++];
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
	
        // Convert all Psi4 C Matrices to einsums Tensor<double, 2>
        ( order[i] == 'C' ) ? nmo1 = ncabs_ : ( order[i] == 'O' ) ? nmo1 = nobs_ : nmo1 = nocc_;
        (order[i+1] == 'C') ? nmo2 = ncabs_ : (order[i+1] == 'O') ? nmo2 = nobs_ : nmo2 = nocc_;
        (order[i+2] == 'C') ? nmo3 = ncabs_ : (order[i+2] == 'O') ? nmo3 = nobs_ : nmo3 = nocc_;
        (order[i+3] == 'C') ? nmo4 = ncabs_ : (order[i+3] == 'O') ? nmo4 = nobs_ : nmo4 = nocc_;

        auto C1 = std::make_unique<Tensor<double, 2>>("C1", nbf1, nmo1);
        auto C2 = std::make_unique<Tensor<double, 2>>("C2", nbf2, nmo2);
        auto C3 = std::make_unique<Tensor<double, 2>>("C3", nbf3, nmo3);
        auto C4 = std::make_unique<Tensor<double, 2>>("C4", nbf4, nmo4);
        {
            convert_C(C1.get(), bs_[o1], nbf1, nmo1);
            convert_C(C2.get(), bs_[o2], nbf2, nmo2);
            convert_C(C3.get(), bs_[o3], nbf3, nmo3);
            convert_C(C4.get(), bs_[o4], nbf4, nmo4);
        }
	
        // Transform ERI AO Tensor to ERI MO Tensor
        auto PQRS = std::make_unique<Tensor<double, 4>>("PQRS", nmo1, nmo2, nmo3, nmo4);
        auto RSPQ = std::make_unique<Tensor<double, 4>>("RSPQ", 0, 0, 0, 0);
        {
            // C1
            auto Pqrs = std::make_unique<Tensor<double, 4>>("Pqrs", nmo1, nbf2, nbf3, nbf4);
            einsum(Indices{P, q, r, s}, &Pqrs, Indices{p, q, r, s}, GAO, Indices{p, P}, C1);
            GAO.reset();
            C1.reset();

            // C3
            auto rsPq = std::make_unique<Tensor<double, 4>>("rsPq", nbf3, nbf4, nmo1, nbf2);
            sort(Indices{r, s, P, q}, &rsPq, Indices{P, q, r, s}, Pqrs);
            Pqrs.reset();
            auto RsPq = std::make_unique<Tensor<double, 4>>("RsPq", nmo3, nbf4, nmo1, nbf2);
            einsum(Indices{R, s, P, q}, &RsPq, Indices{r, s, P, q}, rsPq, Indices{r, R}, C3);
            rsPq.reset();
            C3.reset();

            // C2
            auto RsPQ = std::make_unique<Tensor<double, 4>>("RsPQ", nmo3, nbf4, nmo1, nmo2);
            einsum(Indices{R, s, P, Q}, &RsPQ, Indices{R, s, P, q}, RsPq, Indices{q, Q}, C2);
            RsPq.reset();
            C2.reset();

            // C4
            auto PQRs = std::make_unique<Tensor<double, 4>>("PQRs", nmo1, nmo2, nmo3, nbf4);
            sort(Indices{P, Q, R, s}, &PQRs, Indices{R, s, P, Q}, RsPQ);
            RsPQ.reset();
            einsum(Indices{P, Q, R, index::S}, &PQRS, Indices{P, Q, R, s}, PQRs, Indices{s, index::S}, C4);
            PQRs.reset();
            C4.reset();

            if (nbf3 != nbf1 && nbf3 != nbf2 && nbf3 != nbf4 && int_type == "J") {
                RSPQ = std::make_unique<Tensor<double, 4>>("RSPQ", nmo3, nmo4, nmo1, nmo2);
                sort(Indices{R, index::S, P, Q}, &RSPQ, Indices{P, Q, R, index::S}, PQRS);
            }
        }

        // Stitch into ERI Tensor
        {
            (o1 == 1) ? I = nobs_ : I = 0;
            (o2 == 1) ? J = nobs_ : J = 0;
            (o3 == 1) ? K = nobs_ : K = 0;
            (o4 == 1) ? L = nobs_ : L = 0;

            TensorView<double, 4> ERI_PQRS{*ERI, Dim<4>{nmo1, nmo2, nmo3, nmo4}, Offset<4>{I, J, K, L}};
            set_ERI(ERI_PQRS, PQRS.get());

            if (nbf4 != nbf1 && nbf4 != nbf2 && nbf4 != nbf3 && int_type == "F") {
                Tensor<double, 4> QPSR{"QPSR", nmo2, nmo1, nmo4, nmo3};
                sort(Indices{Q, P, index::S, R}, &QPSR, Indices{P, Q, R, index::S}, PQRS);
                TensorView<double, 4> ERI_QPSR{*ERI, Dim<4>{nmo2, nmo1, nmo4, nmo3}, Offset<4>{J, I, L, K}};
                set_ERI(ERI_QPSR, &QPSR);
            } // end of if statement

            if (nbf3 != nbf1 && nbf3 != nbf2 && nbf3 != nbf4 && int_type == "J") {
                TensorView<double, 4> ERI_RSPQ{*ERI, Dim<4>{nmo3, nmo4, nmo1, nmo2}, Offset<4>{K, L, I, J}};
                set_ERI(ERI_RSPQ, &(*RSPQ));
            } // end of if statement

            if (nbf4 != nbf1 && nbf4 != nbf2 && nbf4 != nbf3 && int_type == "K") {
                Tensor<double, 4> SRQP{"SRQP", nmo4, nmo3, nmo2, nmo1};
                sort(Indices{index::S, R, Q, P}, &SRQP, Indices{P, Q, R, index::S}, PQRS);
                TensorView<double, 4> ERI_SRQP{*ERI, Dim<4>{nmo4, nmo3, nmo2, nmo1}, Offset<4>{L, K, J, I}};
                set_ERI(ERI_SRQP, &SRQP);
            } // end of if statement
        }
    } // end of for loop
}

void MP2F12::form_metric_ints(einsums::Tensor<double, 3> *DF_ERI)
{
    using namespace einsums;
    using namespace tensor_algebra;
    using namespace tensor_algebra::index;

    std::shared_ptr<BasisSet> zero(BasisSet::zero_ao_basis_set());

    std::vector<int> order = {'O', 'O',
                              'O', 'C',
                              'C', 'O',
                              'C', 'C'};
    
    int nmo1, nmo2, R, S, o1, o2;
    for (int idx = 0; idx < (order.size()/2); idx++) {
        int i = idx * 2;
        ( order[i]  == 'C') ? o1 = 1 : o1 = 0;
        (order[i+1] == 'C') ? o2 = 1 : o2 = 0;
        auto bs1 = bs_[o1].basisset();
        auto bs2 = bs_[o2].basisset();
        auto nbf1 = bs1->nbf();
        auto nbf2 = bs2->nbf();

        auto Bpq = std::make_unique<Tensor<double, 3>>("Metric AO", naux_, nbf1, nbf2);
        {
            std::shared_ptr<IntegralFactory> intf(new IntegralFactory(DFBS_, zero, bs1, bs2));

            std::vector<std::shared_ptr<TwoBodyAOInt>> ints;
            for (size_t thread = 0; thread < nthreads_; thread++) {
                ints.push_back(std::shared_ptr<TwoBodyAOInt>(intf->eri()));
            }

#pragma omp parallel for collapse(3) schedule(guided) num_threads(nthreads_)
            for (size_t B = 0; B < DFBS_->nshell(); B++) {
                for (size_t P = 0; P < bs1->nshell(); P++) {
                    for (size_t Q = 0; Q < bs2->nshell(); Q++) {
                        size_t rank = 0;
#ifdef _OPENMP
                        rank = omp_get_thread_num();
#endif
                        const size_t numB = DFBS_->shell(B).nfunction();
                        const size_t numP = bs1->shell(P).nfunction();
                        const size_t numQ = bs2->shell(Q).nfunction();
                        const size_t index_B = DFBS_->shell(B).function_index();
                        const size_t index_P = bs1->shell(P).function_index();
                        const size_t index_Q = bs2->shell(Q).function_index();

                        ints[rank]->compute_shell(B, 0, P, Q);
                        const auto *ints_buff = ints[rank]->buffers()[0];

                        size_t index = 0;
                        for (size_t b = 0; b < numB; b++) {
                            for (size_t p = 0; p < numP; p++) {
                                for (size_t q = 0; q < numQ; q++) {
                                    (*Bpq)(index_B + b,
                                           index_P + p,
                                           index_Q + q) = ints_buff[index++];
                                }
                            }
                        }
                    }
                }
            }
        }

        ( order[i] == 'C' ) ? nmo1 = ncabs_ : nmo1 = nobs_;
        (order[i+1] == 'C') ? nmo2 = ncabs_ : nmo2 = nobs_;

        auto C1 = std::make_unique<Tensor<double, 2>>("C1", nbf1, nmo1);
        auto C2 = std::make_unique<Tensor<double, 2>>("C2", nbf2, nmo2);
        {
            convert_C(C1.get(), bs_[o1], nbf1, nmo1);
            convert_C(C2.get(), bs_[o2], nbf2, nmo2);
        }

        auto BPQ = std::make_unique<Tensor<double, 3>>("BPQ", naux_, nmo1, nmo2);
        {
            // C2
            Tensor<double, 3> BpQ{"BpQ", naux_, nbf1, nmo2};
            einsum(Indices{B, p, Q}, &BpQ, Indices{B, p, q}, Bpq, Indices{q, Q}, C2);
            C2.reset();

            // C1
            Tensor<double, 3> BQp{"BQp", naux_, nmo2, nbf1};
            sort(Indices{B, Q, p}, &BQp, Indices{B, p, Q}, BpQ);
            Tensor<double, 3> BQP{"BQP", naux_, nmo2, nmo1};
            einsum(Indices{B, Q, P}, &BQP, Indices{B, Q, p}, BQp, Indices{p, P}, C1);
            C1.reset();
            sort(Indices{B, P, Q}, &BPQ, Indices{B, Q, P}, BQP);
        }

        auto APQ = std::make_unique<Tensor<double, 3>>("APQ", naux_, nmo1, nmo2);
        {
            auto metric = std::make_shared<FittingMetric>(DFBS_, true);
            metric->form_full_eig_inverse(1.0e-12);
            SharedMatrix Jm12 = metric->get_metric();

            Tensor<double, 2> AB{"JinvAB", naux_, naux_};
#pragma omp parallel for num_threads(nthreads_)
            for (size_t A = 0; A < naux_; A++) {
                for (size_t B = 0; B < naux_; B++) {
                    AB(A, B) = Jm12->get(A, B);
                }
            }

            einsum(Indices{A, P, Q}, &APQ, Indices{A, B}, AB, Indices{B, P, Q}, BPQ);
        }
        BPQ.reset();

        {
            (o1 == 1) ? R = nobs_ : R = 0;
            (o2 == 1) ? S = nobs_ : S = 0;

            TensorView<double, 3> ERI_APQ{*DF_ERI, Dim<3>{naux_, nmo1, nmo2}, Offset<3>{0, R, S}};
            set_ERI(ERI_APQ, APQ.get());
        }
    } // end of for loop
}

void MP2F12::form_oper_ints(const std::string& int_type, einsums::Tensor<double, 3> *DF_ERI)
{
    using namespace einsums;
    using namespace tensor_algebra;
    using namespace tensor_algebra::index;

    std::shared_ptr<BasisSet> zero(BasisSet::zero_ao_basis_set());

    std::vector<int> order = {'o', 'o'};
    if ( int_type != "Uf" || int_type != "FG" ) {
        order = {'o', 'O',
                 'o', 'C'};
    }

    int nmo1, nmo2, R, S, o1, o2;
    for (int idx = 0; idx < (order.size()/2); idx++) {
        int i = idx * 2;
        ( order[i]  == 'C') ? o1 = 1 : o1 = 0;
        (order[i+1] == 'C') ? o2 = 1 : o2 = 0;
        auto bs1 = bs_[o1].basisset();
        auto bs2 = bs_[o2].basisset();
        auto nbf1 = bs1->nbf();
        auto nbf2 = bs2->nbf();

        auto Bpq = std::make_unique<Tensor<double, 3>>("(B|R|pq) AO", naux_, nbf1, nbf2);
        {
            std::shared_ptr<IntegralFactory> intf(
                    new IntegralFactory(DFBS_, zero, bs1, bs2));

            std::vector<std::shared_ptr<TwoBodyAOInt>> ints;
            if ( int_type == "F" ){
                ints.push_back(std::shared_ptr<TwoBodyAOInt>(intf->f12(cgtg_)));
            } else if ( int_type == "FG" ){
                ints.push_back(std::shared_ptr<TwoBodyAOInt>(intf->f12g12(cgtg_)));
            } else if ( int_type == "F2" ){
                ints.push_back(std::shared_ptr<TwoBodyAOInt>(intf->f12_squared(cgtg_)));
            } else if ( int_type == "Uf" ){
                ints.push_back(std::shared_ptr<TwoBodyAOInt>(intf->f12_double_commutator(cgtg_)));
            } else {
                ints.push_back(std::shared_ptr<TwoBodyAOInt>(intf->eri()));
            }

            // Make ints vectors
            for (size_t thread = 1; thread < nthreads_; thread++) {
                ints.push_back(std::shared_ptr<TwoBodyAOInt>(ints[0]->clone()));
            }

#pragma omp parallel for collapse(3) schedule(guided) num_threads(nthreads_)
            for (size_t B = 0; B < DFBS_->nshell(); B++) {
                for (size_t P = 0; P < bs1->nshell(); P++) {
                    for (size_t Q = 0; Q < bs2->nshell(); Q++) {
                        size_t rank = 0;
#ifdef _OPENMP
                        rank = omp_get_thread_num();
#endif
                        const size_t numB = DFBS_->shell(B).nfunction();
                        const size_t numP = bs1->shell(P).nfunction();
                        const size_t numQ = bs2->shell(Q).nfunction();
                        const size_t index_B = DFBS_->shell(B).function_index();
                        const size_t index_P = bs1->shell(P).function_index();
                        const size_t index_Q = bs2->shell(Q).function_index();

                        ints[rank]->compute_shell(B, 0, P, Q);
                        const auto *ints_buff = ints[rank]->buffers()[0];

                        size_t index = 0;
                        for (size_t b = 0; b < numB; b++) {
                            for (size_t p = 0; p < numP; p++) {
                                for (size_t q = 0; q < numQ; q++) {
                                    (*Bpq)(index_B + b,
                                           index_P + p,
                                           index_Q + q) = ints_buff[index++];
                                }
                            }
                        }
                    }
                }
            }
        }

        // Convert all Psi4 C Matrices to einsums Tensor<double, 2>
        ( order[i] == 'C' ) ? nmo1 = ncabs_ : ( order[i] == 'O' ) ? nmo1 = nobs_ : nmo1 = nocc_;
        (order[i+1] == 'C') ? nmo2 = ncabs_ : (order[i+1] == 'O') ? nmo2 = nobs_ : nmo2 = nocc_;

        auto C1 = std::make_unique<Tensor<double, 2>>("C1", nbf1, nmo1);
        auto C2 = std::make_unique<Tensor<double, 2>>("C2", nbf2, nmo2);
        {
            convert_C(C1.get(), bs_[o1], nbf1, nmo1);
            convert_C(C2.get(), bs_[o2], nbf2, nmo2);
        }

        auto BPQ = std::make_unique<Tensor<double, 3>>("BPQ", naux_, nmo1, nmo2);
        {
            // C2
            Tensor<double, 3> BpQ{"BpQ", naux_, nbf1, nmo2};
            einsum(Indices{B, p, Q}, &BpQ, Indices{B, p, q}, Bpq, Indices{q, Q}, C2);
            C2.reset();

            // C1
            Tensor<double, 3> BQp{"BQp", naux_, nmo2, nbf1};
            sort(Indices{B, Q, p}, &BQp, Indices{B, p, Q}, BpQ);
            Tensor<double, 3> BQP{"BQP", naux_, nmo2, nmo1};
            einsum(Indices{B, Q, P}, &BQP, Indices{B, Q, p}, BQp, Indices{p, P}, C1);
            C1.reset();
            sort(Indices{B, P, Q}, &BPQ, Indices{B, Q, P}, BQP);
        }

        {
            (o1 == 1) ? R = nobs_ : R = 0;
            (o2 == 1) ? S = nobs_ : S = 0;

            TensorView<double, 3> ERI_BPQ{*DF_ERI, Dim<3>{naux_, nmo1, nmo2}, Offset<3>{0, R, S}};
            set_ERI(ERI_BPQ, BPQ.get());
        }
    } // end of for loop
}

void MP2F12::form_oper_ints(const std::string& int_type, einsums::Tensor<double, 2> *DF_ERI)
{
    using namespace einsums;
    using namespace tensor_algebra;
    using namespace tensor_algebra::index;

    std::shared_ptr<BasisSet> zero(BasisSet::zero_ao_basis_set());

    std::shared_ptr<IntegralFactory> intf(new IntegralFactory(DFBS_, zero, DFBS_, zero));

    std::vector<std::shared_ptr<TwoBodyAOInt>> ints;
    if ( int_type == "F" ){
        ints.push_back(std::shared_ptr<TwoBodyAOInt>(intf->f12(cgtg_)));
    } else if ( int_type == "FG" ){
        ints.push_back(std::shared_ptr<TwoBodyAOInt>(intf->f12g12(cgtg_)));
    } else if ( int_type == "F2" ){
        ints.push_back(std::shared_ptr<TwoBodyAOInt>(intf->f12_squared(cgtg_)));
    } else if ( int_type == "Uf" ){
        ints.push_back(std::shared_ptr<TwoBodyAOInt>(intf->f12_double_commutator(cgtg_)));
    } else {
        ints.push_back(std::shared_ptr<TwoBodyAOInt>(intf->eri()));
    }

    // Make ints vectors
    for (size_t thread = 1; thread < nthreads_; thread++) {
        ints.push_back(std::shared_ptr<TwoBodyAOInt>(ints[0]->clone()));
    }

#pragma omp parallel for collapse(2) schedule(guided) num_threads(nthreads_)
    for (size_t A = 0; A < DFBS_->nshell(); A++) {
        for (size_t B = 0; B < DFBS_->nshell(); B++) {
            size_t rank = 0;
#ifdef _OPENMP
            rank = omp_get_thread_num();
#endif
            const size_t numA = DFBS_->shell(A).nfunction();
            const size_t numB = DFBS_->shell(B).nfunction();
            const size_t index_A = DFBS_->shell(A).function_index();
            const size_t index_B = DFBS_->shell(B).function_index();

            ints[rank]->compute_shell(A, 0, B, 0);
            const auto *ints_buff = ints[rank]->buffers()[0];

            size_t index = 0;
            for (size_t a = 0; a < numA; a++) {
                for (size_t b = 0; b < numB; b++) {
                    (*DF_ERI)(index_A + a,
                          index_B + b) = ints_buff[index++];
                }
            }
        }
    }
}

void MP2F12::form_df_teints(const std::string& int_type, einsums::Tensor<double, 4> *ERI, einsums::Tensor<double, 3> *Metric)
{
    using namespace einsums;
    using namespace tensor_algebra;
    using namespace tensor_algebra::index;

    auto ARB = std::make_unique<Tensor<double, 2>>("(A|R|B) MO", naux_, naux_);
    form_oper_ints(int_type, ARB.get());

    auto ARPQ = std::make_unique<Tensor<double, 3>>("(A|R|PQ) MO", naux_, nocc_, nri_);
    form_oper_ints(int_type, ARPQ.get());

    // In (PQ|RS) ordering
    std::vector<char> order = {'o', 'o', 'o', 'o'};
    if ( int_type == "G" ) {
        order = {'o', 'O', 'o', 'O',
                 'o', 'O', 'o', 'C'};
    } else if ( int_type == "F" ) {
        order = {'o', 'O', 'o', 'O',
                 'o', 'O', 'o', 'C',
                 'o', 'C', 'o', 'O',
                 'o', 'C', 'o', 'C',};
    } else if ( int_type == "F2" ) {
        order = {'o', 'o', 'o', 'O',
                 'o', 'o', 'o', 'C',};
    }

    int P, Q, R, S;
    int nmo1, nmo2, nmo3, nmo4;
    for (int idx = 0; idx < (order.size()/4); idx++) {
        int i = idx * 4;
        ( order[i] == 'C' ) ? (P = nobs_, nmo1 = ncabs_) :
                            ( order[i] == 'O' ) ? (P = 0, nmo1 = nobs_) : (P = 0, nmo1 = nocc_);
        (order[i+1] == 'C') ? (Q = nobs_, nmo2 = ncabs_) :
                            (order[i+1] == 'O') ? (Q = 0, nmo2 = nobs_) : (Q = 0, nmo2 = nocc_);
        (order[i+2] == 'C') ? (R = nobs_, nmo3 = ncabs_) :
                            (order[i+2] == 'O') ? (R = 0, nmo3 = nobs_) : (R = 0, nmo3 = nocc_);
        (order[i+3] == 'C') ? (S = nobs_, nmo4 = ncabs_) :
                            (order[i+3] == 'O') ? (S = 0, nmo4 = nobs_) : (S = 0, nmo4 = nocc_);

        auto phys_robust = std::make_unique<Tensor<double, 4>>("<PR|F12|QS> MO", nmo1, nmo3, nmo2, nmo4);
        {
            Tensor<double, 4> chem_robust("(PQ|F12|RS) MO", nmo1, nmo2, nmo3, nmo4);

            // Term 1
            Tensor left_metric  = (*Metric)(All, Range{P, nmo1 + P}, Range{Q, nmo2 + Q});
            Tensor right_oper = (*ARPQ)(All, Range{R, nmo3 + R}, Range{S, nmo4 + S});
            einsum(Indices{p, q, r, s}, &chem_robust, Indices{A, p, q}, left_metric, Indices{A, r, s}, right_oper);

            if ( int_type != "G" ) {
                // Term 2
                Tensor right_metric = (*Metric)(All, Range{R, nmo3 + R}, Range{S, nmo4 + S});
                Tensor left_oper  = (*ARPQ)(All, Range{P, nmo1 + P}, Range{Q, nmo2 + Q});
                einsum(1.0, Indices{p, q, r, s}, &chem_robust,
                       1.0, Indices{A, p, q}, left_oper, Indices{A, r, s}, right_metric);

                // Term 3
                Tensor<double, 3> tmp{"Temp", naux_, nmo3, nmo4};
                einsum(Indices{A, r, s}, &tmp, Indices{A, B}, ARB, Indices{B, r, s}, right_metric);
                einsum(1.0, Indices{p, q, r, s}, &chem_robust,
                       -1.0, Indices{A, p, q}, left_metric, Indices{A, r, s}, tmp);
            }

            sort(Indices{p, r, q, s}, &phys_robust, Indices{p, q, r, s}, chem_robust);
        }

        {
            TensorView<double, 4> ERI_PRQS{(*ERI), Dim<4>{nmo1, nmo3, nmo2, nmo4}, Offset<4>{P, R, Q, S}};
            set_ERI(ERI_PRQS, phys_robust.get());
        }
    } // end of for loop
}

}} // End namespaces
