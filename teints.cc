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
#include "psi4/libpsi4util/process.h"
#endif

#include "teints.h"

#include "psi4/libmints/basisset.h"
#include "psi4/libmints/integral.h"
#include "psi4/libmints/integralparameters.h"
#include "psi4/libmints/matrix.h"
#include "psi4/libmints/orbitalspace.h"
#include "psi4/lib3index/dftensor.h"

#include "einsums/TensorAlgebra.hpp"
#include "einsums/Sort.hpp"
#include "einsums/Timer.hpp"
#include "einsums/Tensor.hpp"
#include "einsums/Print.hpp"

namespace psi{ namespace MP2F12 {

void teints(const std::string& int_type, einsums::Tensor<double, 4> *ERI, std::vector<OrbitalSpace>& bs, 
            const int& nobs, std::shared_ptr<CorrelationFactor> corr)
{
    using namespace einsums; 
    using namespace tensor_algebra;
    using namespace tensor_algebra::index;

    int nthreads = 1;
#ifdef _OPENMP
    nthreads = Process::environment.get_n_threads();
#endif

    std::vector<int> o_tei = {0, 0, 0, 0};
    if ( int_type == "F" ){
        o_tei = {0, 0, 0, 0,
                 0, 0, 0, 1,
                 0, 0, 1, 1};
    } else if ( int_type == "F2" ){
        o_tei = {0, 0, 0, 0,
                 0, 0, 0, 1};
    } else if ( int_type == "G" ){ 
        o_tei = {0, 0, 0, 0,
                 0, 0, 0, 1,
                 1, 0, 0, 1,
                 1, 0, 1, 0}; 
    }

    int nmo1, nmo2, nmo3, nmo4;
    int I, J, K, L;
    for (int idx = 0; idx < (o_tei.size()/4); idx++) {
        int i = idx * 4;
        auto bs1 = bs[o_tei[i]].basisset();
        auto bs2 = bs[o_tei[i+1]].basisset();
        auto bs3 = bs[o_tei[i+2]].basisset();
        auto bs4 = bs[o_tei[i+3]].basisset();
        auto nbf1 = bs1->nbf();
        auto nbf2 = bs2->nbf();
        auto nbf3 = bs3->nbf();
        auto nbf4 = bs4->nbf();

        // Create ERI AO Tensor
        timer::push("ERI Building");
        auto GAO = std::make_unique<Tensor<double, 4>>("ERI AO", nbf1, nbf2, nbf3, nbf4);
        {
            IntegralFactory intf(bs1, bs3, bs2, bs4);

            auto ints = std::shared_ptr<TwoBodyAOInt>(nullptr);
            if ( int_type == "F" ){
                ints = std::shared_ptr<TwoBodyAOInt>(intf.f12(corr));
            } else if ( int_type == "FG" ){
                ints = std::shared_ptr<TwoBodyAOInt>(intf.f12g12(corr));
            } else if ( int_type == "F2" ){
                ints = std::shared_ptr<TwoBodyAOInt>(intf.f12_squared(corr));
            } else if ( int_type == "Uf" ){
                ints = std::shared_ptr<TwoBodyAOInt>(intf.f12_double_commutator(corr));
            } else { 
                ints = std::shared_ptr<TwoBodyAOInt>(intf.eri());
            }

            const double *buffer = ints->buffer();
            for (int M = 0; M < bs1->nshell(); M++) {
                for (int N = 0; N < bs3->nshell(); N++) {
                    for (int P = 0; P < bs2->nshell(); P++) {
                        for (int Q = 0; Q < bs4->nshell(); Q++) {
                            ints->compute_shell(M, N, P, Q);
                            int mM = bs1->shell(M).nfunction();
                            int nN = bs3->shell(N).nfunction();
                            int pP = bs2->shell(P).nfunction();
                            int qQ = bs4->shell(Q).nfunction();

                            #pragma omp parallel for collapse(4) num_threads(nthreads)
                            for (int m = 0; m < mM; m++) {
                                for (int n = 0; n < nN; n++) {
                                    for (int p = 0; p < pP; p++) {
                                        for (int q = 0; q < qQ; q++) {
                                            int index = q + qQ * (p + pP * (n + nN * m));
                                            (*GAO)(bs1->shell(M).function_index() + m,
                                                   bs2->shell(P).function_index() + p,
                                                   bs3->shell(N).function_index() + n, 
                                                   bs4->shell(Q).function_index() + q) = buffer[index];
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        timer::pop(); // ERI Building
	
        // Convert all Psi4 C Matrices to einsums Tensor<double, 2>
        timer::push("Convert C Matrices to Tensors");
        (o_tei[i] == 1) ? nmo1 = nbf1 - nobs : nmo1 = nbf1;
        (o_tei[i+1] == 1) ? nmo2 = nbf2 - nobs : nmo2 = nbf2;
        (o_tei[i+2] == 1) ? nmo3 = nbf3 - nobs : nmo3 = nbf3;
        (o_tei[i+3] == 1) ? nmo4 = nbf4 - nobs : nmo4 = nbf4;

        auto C1 = std::make_unique<Tensor<double, 2>>("C1", nbf1, nmo1);
        auto C2 = std::make_unique<Tensor<double, 2>>("C2", nbf2, nmo2);
        auto C3 = std::make_unique<Tensor<double, 2>>("C3", nbf3, nmo3);
        auto C4 = std::make_unique<Tensor<double, 2>>("C4", nbf4, nmo4);
        {
            #pragma omp parallel for collapse(2) num_threads(nthreads)
            for (int p = 0; p < nbf1; p++) {
                for (int q = 0; q < nmo1; q++) {
                    (*C1)(p,q) = bs[o_tei[i]].C()->get(p,q);	
                }
            }

            #pragma omp parallel for collapse(2) num_threads(nthreads)
            for (int p = 0; p < nbf2; p++) {
                for (int q = 0; q < nmo2; q++) {
                    (*C2)(p,q) = bs[o_tei[i+1]].C()->get(p,q);	
                }
            }

            #pragma omp parallel for collapse(2) num_threads(nthreads)
            for (int p = 0; p < nbf3; p++) {
                for (int q = 0; q < nmo3; q++) {
                    (*C3)(p,q) = bs[o_tei[i+2]].C()->get(p,q);	
                }
            }

            #pragma omp parallel for collapse(2) num_threads(nthreads)
            for (int p = 0; p < nbf4; p++) {
                for (int q = 0; q < nmo4; q++) {
                    (*C4)(p,q) = bs[o_tei[i+3]].C()->get(p,q);	
                }
            }
        }
        timer::pop(); // Convert C Matrices to Tensors
	
        // Transform ERI AO Tensor to ERI MO Tensor
        timer::push("Full Transformation");
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
            einsum(Indices{P, Q, R, S}, &PQRS, Indices{P, Q, R, s}, PQRs, Indices{s, S}, C4);
            PQRs.reset();
            C4.reset();

            if (nbf4 != nbf1 && nbf4 != nbf2 && nbf4 != nbf3 && int_type != "F2") {
                RSPQ = std::make_unique<Tensor<double, 4>>("RSPQ", nmo3, nmo4, nmo1, nmo2);
                sort(Indices{R, S, P, Q}, &RSPQ, Indices{P, Q, R, S}, PQRS);
            }
        }
        timer::pop(); // Full Transformation

        // Stitch into ERI Tensor
        timer::push("Stitch into ERI Tensor");
        {
            (o_tei[i] == 1) ? I = nobs : I = 0;
            (o_tei[i+1] == 1) ? J = nobs : J = 0;
            (o_tei[i+2] == 1) ? K = nobs : K = 0;
            (o_tei[i+3] == 1) ? L = nobs : L = 0;

            timer::push("Put into ERI Tensor");
            TensorView<double, 4> ERI_PQRS{*ERI, Dim<4>{nmo1, nmo2, nmo3, nmo4}, Offset<4>{I, J, K, L}};
            #pragma omp parallel for collapse(4) num_threads(nthreads)
            for (int i = 0; i < nmo1; i++){
                for (int j = 0; j < nmo2; j++){
                    for (int k = 0; k < nmo3; k++){
                        for (int l = 0; l < nmo4; l++){
                            ERI_PQRS(i, j, k, l) = (*PQRS)(i, j, k, l);
                        }
                    }
                }
            }	
            timer::pop();

            if (nbf4 != nbf1 && nbf4 != nbf2 && nbf4 != nbf3 && int_type != "F2") {
                Tensor<double, 4> QPSR{"QPSR", nmo2, nmo1, nmo4, nmo3};
                sort(Indices{Q, P, S, R}, &QPSR, Indices{R, S, P, Q}, RSPQ);
                timer::push("Put into ERI Tensor");
                TensorView<double, 4> ERI_QPSR{*ERI, Dim<4>{nmo2, nmo1, nmo4, nmo3}, Offset<4>{J, I, L, K}};
                #pragma omp parallel for collapse(4) num_threads(nthreads)
                for (int j = 0; j < nmo2; j++){
                    for (int i = 0; i < nmo1; i++){
                        for (int l = 0; l < nmo4; l++){
                            for (int k = 0; k < nmo3; k++){
                                ERI_QPSR(j, i, l, k) = QPSR(j, i, l, k);
                            }
                        }
                    }
                }
                timer::pop();
            } // end of if statement

            if (nbf4 != nbf1 && nbf4 != nbf2 && nbf4 != nbf3 && int_type == "G") {
                Tensor<double, 4> SRQP{"SRQP", nmo4, nmo3, nmo2, nmo1};
                sort(Indices{S, R, Q, P}, &SRQP, Indices{P, Q, R, S}, PQRS);
                timer::push("Put into ERI Tensor");
                TensorView<double, 4> ERI_SRQP{*ERI, Dim<4>{nmo4, nmo3, nmo2, nmo1}, Offset<4>{L, K, J, I}};
                #pragma omp parallel for collapse(4) num_threads(nthreads)
                for (int l = 0; l < nmo4; l++){
                    for (int k = 0; k < nmo3; k++){
                        for (int j = 0; j < nmo2; j++){
                            for (int i = 0; i < nmo1; i++){
                                ERI_SRQP(l, k, j, i) = SRQP(l, k, j, i);
                            }
                        }
                    }
                }
                timer::pop();
            } // end of if statement
        }
        RSPQ.reset(nullptr);
        PQRS.reset(nullptr);
        timer::pop(); // Stitch into ERI Tensor
    } // end of for loop
}

void metric_ints(einsums::Tensor<double, 3> *DF_ERI, const std::vector<OrbitalSpace>& bs,
                 const std::shared_ptr<BasisSet> dfbs, const int& nobs, const int& naux, const int& nri)
{
    using namespace einsums;
    using namespace tensor_algebra;
    using namespace tensor_algebra::index;
    
    int nthreads = 1;
#ifdef _OPENMP
    nthreads = Process::environment.get_n_threads();
#endif

    std::shared_ptr<BasisSet> zero(BasisSet::zero_ao_basis_set());

    std::vector<int> o_ints = {0, 0,
                               0, 1,
                               1, 0,
                               1, 1};
    
    int nmo1, nmo2;
    int R, S;
    for (int idx = 0; idx < (o_ints.size()/2); idx++) {
        int i = idx * 2;
        std::shared_ptr<BasisSet> bs1(bs[o_ints[i]].basisset());
        std::shared_ptr<BasisSet> bs2(bs[o_ints[i+1]].basisset());
        auto nbf1 = bs1->nbf();
        auto nbf2 = bs2->nbf();

        timer::push("AO Building");
        auto Bpq = std::make_unique<Tensor<double, 3>>("Metric AO", naux, nbf1, nbf2);
        {
            std::shared_ptr<IntegralFactory> intf(new IntegralFactory(dfbs, zero, bs1, bs2));
            auto ints = std::shared_ptr<TwoBodyAOInt>(intf->eri());

            const double *buffer = ints->buffer();
            for (int B = 0; B < dfbs->nshell(); B++) {
                for (int P = 0; P < bs1->nshell(); P++) {
                    for (int Q = 0; Q < bs2->nshell(); Q++) {
                        int numB = dfbs->shell(B).nfunction();
                        int numP = bs1->shell(P).nfunction();
                        int numQ = bs2->shell(Q).nfunction();
                        int Bstart = dfbs->shell(B).function_index();
                        int Pstart = bs1->shell(P).function_index();
                        int Qstart = bs2->shell(Q).function_index();

                        ints->compute_shell(B, 0, P, Q);

                        #pragma omp parallel for collapse(3) num_threads(nthreads)
                        for (int b = 0; b < numB; b++) {
                            for (int p = 0; p < numP; p++) {
                                for (int q = 0; q < numQ; q++) {
                                    int index = q + numQ * (p + numP * b);
                                    (*Bpq)(Bstart + b,
                                           Pstart + p,
                                           Qstart + q) = buffer[index];
                                }
                            }
                        }
                    }
                }
            }
        }
        timer::pop(); // AO Building

        timer::push("Convert C Matrices to Tensors");
        (o_ints[i] == 1) ? nmo1 = nbf1 - nobs : nmo1 = nbf1;
        (o_ints[i+1] == 1) ? nmo2 = nbf2 - nobs : nmo2 = nbf2;

        auto C1 = std::make_unique<Tensor<double, 2>>("C1", nbf1, nmo1);
        auto C2 = std::make_unique<Tensor<double, 2>>("C2", nbf2, nmo2);
        {
            #pragma omp parallel for collapse(2) num_threads(nthreads)
            for (int p = 0; p < nbf1; p++) {
                for (int q = 0; q < nmo1; q++) {
                    (*C1)(p, q) = bs[o_ints[i]].C()->get(p, q);
                }
            }

            #pragma omp parallel for collapse(2) num_threads(nthreads)
            for (int p = 0; p < nbf2; p++) {
                for (int q = 0; q < nmo2; q++) {
                    (*C2)(p, q) = bs[o_ints[i+1]].C()->get(p, q);
                }
            }
        }
        timer::pop(); // Convert C Matrices

        timer::push("Full Transformation");
        auto BPQ = std::make_unique<Tensor<double, 3>>("BPQ", naux, nmo1, nmo2);
        {
            // C2
            Tensor<double, 3> BpQ{"BpQ", naux, nbf1, nmo2};
            einsum(Indices{B, p, Q}, &BpQ, Indices{B, p, q}, Bpq, Indices{q, Q}, C2);
            C2.reset();

            // C1
            Tensor<double, 3> BQp{"BQp", naux, nmo2, nbf1};
            sort(Indices{B, Q, p}, &BQp, Indices{B, p, Q}, BpQ);
            Tensor<double, 3> BQP{"BQP", naux, nmo2, nmo1};
            einsum(Indices{B, Q, P}, &BQP, Indices{B, Q, p}, BQp, Indices{p, P}, C1);
            C1.reset();
            sort(Indices{B, P, Q}, &BPQ, Indices{B, Q, P}, BQP);
        }
        timer::pop(); // Full Transform

        timer::push("Form Metric");
        auto APQ = std::make_unique<Tensor<double, 3>>("APQ", naux, nmo1, nmo2);
        {
            auto metric = std::make_shared<FittingMetric>(dfbs, true);
            metric->form_full_eig_inverse(1.0e-12);
            SharedMatrix Jm12 = metric->get_metric();

            Tensor<double, 2> AB{"JinvAB", naux, naux};
            #pragma omp parallel for collapse(2) num_threads(nthreads)
            for (int A = 0; A < naux; A++) {
                for (int B = 0; B < naux; B++) {
                    AB(A, B) = Jm12->get(A, B);
                }
            }

            einsum(Indices{A, P, Q}, &APQ, Indices{A, B}, AB, Indices{B, P, Q}, BPQ);
        }
        BPQ.reset();
        timer::pop(); // Form Metric

        timer::push("Put into DF_ERI Tensor");
        {
            (o_ints[i] == 1) ? R = nobs : R = 0;
            (o_ints[i+1] == 1) ? S = nobs : S = 0;

            TensorView<double, 3> ERI_APQ{*DF_ERI, Dim<3>{naux, nmo1, nmo2}, Offset<3>{0, R, S}};
            #pragma omp parallel for collapse(3) num_threads(nthreads)
            for (int A = 0; A < naux; A++){
                for (int P = 0; P < nmo1; P++){
                    for (int Q = 0; Q < nmo2; Q++){
                        ERI_APQ(A, P, Q) = (*APQ)(A, P, Q);
                    }
                }
            }
        }
        APQ.reset();
        timer::pop(); // Put in to DF_ERI Tensor
    } // end of for loop
}

void oper_ints(const std::string& int_type, einsums::Tensor<double, 3> *DF_ERI, einsums::Tensor<double, 2> *AB,
           std::vector<OrbitalSpace>& bs, std::shared_ptr<BasisSet> dfbs,
           const int& nobs, const int& naux, std::shared_ptr<CorrelationFactor> corr)
{
    using namespace einsums;
    using namespace tensor_algebra;
    using namespace tensor_algebra::index;

    int nthreads = 1;
#ifdef _OPENMP
    nthreads = Process::environment.get_n_threads();
#endif

    std::shared_ptr<BasisSet> zero(BasisSet::zero_ao_basis_set());

    std::vector<int> o_ints = {0, 0};
    if ( int_type == "F" || int_type == "F2" ){
        o_ints = {0, 0,
                  0, 1};
    } else if ( int_type == "G" ){ 
        o_ints = {0, 0,
                  0, 1,
                  1, 0,
                  1, 1}; 
    }

    int nmo1, nmo2;
    int R, S;
    for (int idx = 0; idx < (o_ints.size()/2); idx++) {
        int i = idx * 2;
        std::shared_ptr<BasisSet> bs1(bs[o_ints[i]].basisset());
        std::shared_ptr<BasisSet> bs2(bs[o_ints[i+1]].basisset());
        auto nbf1 = bs1->nbf();
        auto nbf2 = bs2->nbf();

        timer::push("AO Building");
        auto Bpq = std::make_unique<Tensor<double, 3>>("(B|R|pq) AO", naux, nbf1, nbf2);
        {
            std::shared_ptr<IntegralFactory> intf_Bpq(
                    new IntegralFactory(dfbs, zero, bs1, bs2));
            std::shared_ptr<IntegralFactory> intf_AB(
                    new IntegralFactory(dfbs, zero, dfbs, zero));

            auto ints_Bpq = std::shared_ptr<TwoBodyAOInt>(nullptr);
            auto ints_AB = std::shared_ptr<TwoBodyAOInt>(nullptr);
            if ( int_type == "F" ){
                ints_Bpq = std::shared_ptr<TwoBodyAOInt>(intf_Bpq->f12(corr));
                ints_AB = std::shared_ptr<TwoBodyAOInt>(intf_AB->f12(corr));
            } else if ( int_type == "FG" ){
                ints_Bpq = std::shared_ptr<TwoBodyAOInt>(intf_Bpq->f12g12(corr));
                ints_AB = std::shared_ptr<TwoBodyAOInt>(intf_AB->f12g12(corr));
            } else if ( int_type == "F2" ){
                ints_Bpq = std::shared_ptr<TwoBodyAOInt>(intf_Bpq->f12_squared(corr));
                ints_AB = std::shared_ptr<TwoBodyAOInt>(intf_AB->f12_squared(corr));
            } else if ( int_type == "Uf" ){
                ints_Bpq = std::shared_ptr<TwoBodyAOInt>(intf_Bpq->f12_double_commutator(corr));
                ints_AB = std::shared_ptr<TwoBodyAOInt>(intf_AB->f12_double_commutator(corr));
            } else {
                ints_Bpq = std::shared_ptr<TwoBodyAOInt>(intf_Bpq->eri());
                ints_AB = std::shared_ptr<TwoBodyAOInt>(intf_AB->eri());
            }

            timer::push("(B|R|pq)");
            const double *buffer_Bpq = ints_Bpq->buffer();
            for (int B = 0; B < dfbs->nshell(); B++) {
                for (int P = 0; P < bs1->nshell(); P++) {
                    for (int Q = 0; Q < bs2->nshell(); Q++) {
                        int numB = dfbs->shell(B).nfunction();
                        int numP = bs1->shell(P).nfunction();
                        int numQ = bs2->shell(Q).nfunction();
                        int Bstart = dfbs->shell(B).function_index();
                        int Pstart = bs1->shell(P).function_index();
                        int Qstart = bs2->shell(Q).function_index();

                        ints_Bpq->compute_shell(B, 0, P, Q);

                        #pragma omp parallel for collapse(3) num_threads(nthreads)
                        for (int b = 0; b < numB; b++) {
                            for (int p = 0; p < numP; p++) {
                                for (int q = 0; q < numQ; q++) {
                                    int index = q + numQ * (p + numP * b);
                                    (*Bpq)(Bstart + b,
                                           Pstart + p,
                                           Qstart + q) = buffer_Bpq[index];
                                }
                            }
                        }
                    }
                }
            }
            timer::pop();

            timer::push("(A|R|B)");

            const double *buffer_AB = ints_AB->buffer();
            for (int A = 0; A < dfbs->nshell(); A++) {
                for (int B = 0; B < dfbs->nshell(); B++) {
                    int numA = dfbs->shell(A).nfunction();
                    int numB = dfbs->shell(B).nfunction();
                    int Astart = dfbs->shell(A).function_index();
                    int Bstart = dfbs->shell(B).function_index();

                    ints_AB->compute_shell(A, 0, B, 0);

                    int index = 0;
                    #pragma omp parallel for collapse(2) num_threads(nthreads)
                    for (int a = 0; a < numA; a++) {
                        for (int b = 0; b < numB; b++) {
                            int index = b + numB * a;
                            (*AB)(Astart + a,
                                  Bstart + b) = buffer_AB[index];
                        }
                    }
                }
            }
            timer::pop();
        }
        timer::pop(); // AO Building

        // Convert all Psi4 C Matrices to einsums Tensor<double, 2>
        timer::push("Convert C Matrices to Tensors");
        (o_ints[i] == 1) ? nmo1 = nbf1 - nobs : nmo1 = nbf1;
        (o_ints[i+1] == 1) ? nmo2 = nbf2 - nobs : nmo2 = nbf2;

        auto C1 = std::make_unique<Tensor<double, 2>>("C1", nbf1, nmo1);
        auto C2 = std::make_unique<Tensor<double, 2>>("C2", nbf2, nmo2);
        {
            #pragma omp parallel for collapse(2) num_threads(nthreads)
            for (int p = 0; p < nbf1; p++) {
                for (int q = 0; q < nmo1; q++) {
                    (*C1)(p, q) = bs[o_ints[i]].C()->get(p, q);
                }
            }

            #pragma omp parallel for collapse(2) num_threads(nthreads)
            for (int p = 0; p < nbf2; p++) {
                for (int q = 0; q < nmo2; q++) {
                    (*C2)(p, q) = bs[o_ints[i+1]].C()->get(p, q);
                }
            }
        }
        timer::pop(); // Convert C Matrices to Tensors

        timer::push("Full Transformation");
        auto BPQ = std::make_unique<Tensor<double, 3>>("BPQ", naux, nmo1, nmo2);
        {
            // C2
            Tensor<double, 3> BpQ{"BpQ", naux, nbf1, nmo2};
            einsum(Indices{B, p, Q}, &BpQ, Indices{B, p, q}, Bpq, Indices{q, Q}, C2);
            C2.reset();

            // C1
            Tensor<double, 3> BQp{"BQp", naux, nmo2, nbf1};
            sort(Indices{B, Q, p}, &BQp, Indices{B, p, Q}, BpQ);
            Tensor<double, 3> BQP{"BQP", naux, nmo2, nmo1};
            einsum(Indices{B, Q, P}, &BQP, Indices{B, Q, p}, BQp, Indices{p, P}, C1);
            C1.reset();
            sort(Indices{B, P, Q}, &BPQ, Indices{B, Q, P}, BQP);
        }
        timer::pop(); // Full Transform

        timer::push("Put into DF_ERI Tensor");
        {
            (o_ints[i] == 1) ? R = nobs : R = 0;
            (o_ints[i+1] == 1) ? S = nobs : S = 0;

            TensorView<double, 3> ERI_BPQ{*DF_ERI, Dim<3>{naux, nmo1, nmo2}, Offset<3>{0, R, S}};
            #pragma omp parallel for collapse(3) num_threads(nthreads)
            for (int B = 0; B < naux; B++){
                for (int P = 0; P < nmo1; P++){
                    for (int Q = 0; Q < nmo2; Q++){
                        ERI_BPQ(B, P, Q) = (*BPQ)(B, P, Q);
                    }
                }
            }
        }
        BPQ.reset();
        timer::pop(); // Put in to DF_ERI Tensor
    } // end of for loop
}

void df_teints(const std::string& int_type, einsums::Tensor<double, 4> *ERI, einsums::Tensor<double, 3> *Metric,
            std::vector<OrbitalSpace>& bs, std::shared_ptr<BasisSet> dfbs,
            const int& nobs, const int& naux, const int& nri, std::shared_ptr<CorrelationFactor> corr)
{
    using namespace einsums;
    using namespace tensor_algebra;
    using namespace tensor_algebra::index;
    int ncabs = nri - nobs;

    int nthreads = 1;
#ifdef _OPENMP
    nthreads = Process::environment.get_n_threads();
#endif

    auto ARB = std::make_unique<Tensor<double, 2>>("(A|R|B) MO", naux, naux);
    auto ARPQ = std::make_unique<Tensor<double, 3>>("(A|R|PQ) MO", naux, nri, nri);

    oper_ints(int_type, ARPQ.get(), ARB.get(), bs, dfbs, nobs, naux, corr);

    // In (PQ|RS) ordering
    std::vector<int> o_tei = {0, 0, 0, 0};
    if ( int_type == "F" ){
        o_tei = {0, 0, 0, 0,
                 0, 0, 0, 1,
                 0, 1, 0, 0,
                 0, 1, 0, 1};
    } else if ( int_type == "F2" ){
        o_tei = {0, 0, 0, 0,
                 0, 0, 0, 1};
    } else if ( int_type == "G" ){ 
        o_tei = {0, 0, 0, 0,
                 0, 0, 0, 1,
                 0, 1, 0, 0,
                 1, 0, 0, 0,
                 1, 0, 0, 1,
                 1, 1, 0, 0}; 
    }

    int P, Q, R, S;
    int nmo1, nmo2, nmo3, nmo4;
    for (int idx = 0; idx < (o_tei.size()/4); idx++) {
        int i = idx * 4;
        (o_tei[i]  ==  1) ? (P = nobs, nmo1 = ncabs) : (P = 0, nmo1 = nobs);
        (o_tei[i+1] == 1) ? (Q = nobs, nmo2 = ncabs) : (Q = 0, nmo2 = nobs);
        (o_tei[i+2] == 1) ? (R = nobs, nmo3 = ncabs) : (R = 0, nmo3 = nobs);
        (o_tei[i+3] == 1) ? (S = nobs, nmo4 = ncabs) : (S = 0, nmo4 = nobs);

        auto phys_robust = std::make_unique<Tensor<double, 4>>("<PR|F12|QS> MO", nmo1, nmo3, nmo2, nmo4);
        {
            Tensor<double, 4> chem_robust("(PQ|F12|RS) MO", nmo1, nmo2, nmo3, nmo4);
            Tensor left_metric  = (*Metric)(All, Range{P, nmo1 + P}, Range{Q, nmo2 + Q});
            Tensor right_metric = (*Metric)(All, Range{R, nmo3 + R}, Range{S, nmo4 + S});
            Tensor left_oper  = (*ARPQ)(All, Range{P, nmo1 + P}, Range{Q, nmo2 + Q});
            Tensor right_oper = (*ARPQ)(All, Range{R, nmo3 + R}, Range{S, nmo4 + S});

            timer::push("Term 1");
            einsum(Indices{p, q, r, s}, &chem_robust, Indices{A, p, q}, left_metric, Indices{A, r, s}, right_oper);
            timer::pop();

            timer::push("Term 2");
            einsum(1.0, Indices{p, q, r, s}, &chem_robust,
                   1.0, Indices{A, p, q}, left_oper, Indices{A, r, s}, right_metric);
            timer::pop();

            timer::push("Term 3");
            Tensor<double, 3> tmp{"Temp", naux, nmo3, nmo4};
            einsum(Indices{A, r, s}, &tmp, Indices{A, B}, ARB, Indices{B, r, s}, right_metric);
            einsum(1.0, Indices{p, q, r, s}, &chem_robust,
                   -1.0, Indices{A, p, q}, left_metric, Indices{A, r, s}, tmp);
            timer::pop();

            timer::push("Chem to Phys");
            sort(Indices{p, r, q, s}, &phys_robust, Indices{p, q, r, s}, chem_robust);
            timer::pop();
        }

        timer::push("Stitch into ERI Tensor");
        {
            TensorView<double, 4> ERI_PRQS{(*ERI), Dim<4>{nmo1, nmo3, nmo2, nmo4}, Offset<4>{P, R, Q, S}};
            #pragma omp parallel for collapse(4) num_threads(nthreads) schedule(guided)
            for (int p = 0; p < nmo1; p++){
                for (int r = 0; r < nmo3; r++){
                    for (int q = 0; q < nmo2; q++){
                        for (int s = 0; s < nmo4; s++){
                            ERI_PRQS(p, r, q, s) = (*phys_robust)(p, r, q, s);
                        }
                    }
                }
            }
            phys_robust.reset();
        }
        timer::pop(); // Stitch into ERI Tensor
    } // end of for loop
}

}} // End namespaces