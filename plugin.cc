/*
 * @BEGIN LICENSE
 *
 * MP2F12 by Psi4 Developer, a plugin to:
 *
 * Psi4: an open-source quantum chemistry software package
 *
 * Copyright (c) 2007-2021 The Psi4 Developers.
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
#endif

#include "psi4/psi4-dec.h"
#include "psi4/libqt/qt.h"
#include "psi4/libpsi4util/PsiOutStream.h"
#include "psi4/liboptions/liboptions.h"
#include "psi4/libpsio/psio.h"
#include "psi4/libpsio/psio.hpp"
#include "psi4/libpsi4util/process.h"

#include "psi4/libmints/basisset.h"
#include "psi4/libmints/dimension.h"
#include "psi4/libmints/eri.h"
#include "psi4/libmints/integral.h"
#include "psi4/libmints/integralparameters.h"
#include "psi4/libmints/fjt.h"
#include "psi4/libmints/matrix.h"
#include "psi4/libmints/mintshelper.h"
#include "psi4/libmints/orbitalspace.h"
#include "psi4/libmints/twobody.h"
#include "psi4/libmints/wavefunction.h"

#include "einsums/TensorAlgebra.hpp"
#include "einsums/LinearAlgebra.hpp"
#include "einsums/State.hpp"
#include "einsums/Timer.hpp"
#include "einsums/Tensor.hpp"
#include "einsums/Utilities.hpp"
#include "einsums/Print.hpp"
#include "einsums/STL.hpp"

namespace psi{ namespace MP2F12 {

extern "C" PSI_API
int read_options(std::string name, Options& options)
{
    if (name == "MP2F12"|| options.read_globals()) {
        /*- The amount of information printed to the output file -*/
        options.add_int("PRINT", 0);
        options.add_bool("WRITE_INTS", false);
        options.add_bool("READ_INTS", false);
    }

    return true;
}

void print_r2_tensor(einsums::Tensor<double, 2> *M)
{
    using namespace einsums;

    int rows = (*M).dim(0);
    int cols = (*M).dim(1);
    auto M_psi4 = std::make_shared<Matrix>((*M).name(), rows, cols);

#pragma omp for collapse(2)
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            M_psi4->set(i, j, (*M)(i,j));
        }	
    }

    M_psi4->print_to_numpy();
}

void print_r2_tensor(einsums::TensorView<double, 2> M)
{
    using namespace einsums;

    int rows = M.dim(0);
    int cols = M.dim(1);
    auto M_psi4 = std::make_shared<Matrix>(M.name(), rows, cols);

#pragma omp for collapse(2)
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            M_psi4->set(i, j, M(i,j));
        }	
    }

    M_psi4->print_to_numpy();
}

void print_r4_tensor(einsums::Tensor<double, 4> *M)
{
    using namespace einsums;

    int r1 = (*M).dim(0);
    int r2 = (*M).dim(1);
    int c1 = (*M).dim(2);
    int c2 = (*M).dim(3);
    auto M_psi4 = std::make_shared<Matrix>((*M).name(), r1 * r2, c1 * c2);

#pragma omp for collapse(4)
    for (int p = 0; p < r1; p++) {
        for (int q = 0; q < r2; q++) {
            for (int r = 0; r < c1; r++) {
                for (int s = 0; s < c2; s++) {
                    M_psi4->set(p * r2 + q, r * c2 + s, (*M)(p,q,r,s));
                }
            }
        }
    }

    M_psi4->print_to_numpy();
}

void write_ints_disk(einsums::Tensor<double, 2> *ERI, int s1, int s2)
{
    using namespace einsums; 

    println("Writing {} ...", (*ERI).name());
    outfile->Printf("  Writing %20d ...\n", (*ERI).name());
    state::data = h5::open("Data.h5", H5F_ACC_RDWR);
    timer::push((*ERI).name() + " write");
    DiskTensor<double, 2> ERI_(state::data, (*ERI).name(), s1, s2);
    ERI_(All, All) = (*ERI);
    timer::pop();
}

void write_ints_disk(einsums::Tensor<double, 4> *ERI, int s1, int s2, int s3, int s4)
{
    using namespace einsums; 

    println("Writing {} ...", (*ERI).name());
    outfile->Printf("  Writing %20d ...\n", (*ERI).name());
    state::data = h5::open("Data.h5", H5F_ACC_RDWR);
    timer::push((*ERI).name() + " write");
    DiskTensor<double, 4> ERI_(state::data, (*ERI).name(), s1, s2, s3, s4);
    ERI_(All, All, All, All) = (*ERI);
    timer::pop();
}

void read_ints_disk(einsums::Tensor<double, 2> *ERI, int s1, int s2)
{
    using namespace einsums; 

    println("Retrieving {}...", (*ERI).name());
    outfile->Printf("  Retrieving %20d ...\n", (*ERI).name());
    timer::push((*ERI).name() + " read");
    DiskTensor<double, 2> ERI_(state::data, (*ERI).name(), s1, s2);
    auto diskView_ERI = ERI_(All, All);
    *ERI = diskView_ERI.get();
    timer::pop();
}

void read_ints_disk(einsums::Tensor<double, 4> *ERI, int s1, int s2, int s3, int s4)
{
    using namespace einsums; 

    println("Retrieving {}...", (*ERI).name());
    outfile->Printf("  Retrieving %20d ...\n", (*ERI).name());
    timer::push((*ERI).name() + " read");
    DiskTensor<double, 4> ERI_(state::data, (*ERI).name(), s1, s2, s3, s4);
    auto diskView_ERI = ERI_(All, All, All, All);
    *ERI = diskView_ERI.get();
    timer::pop();
}

void oeints(MintsHelper mints, einsums::Tensor<double, 2> *h, 
		    std::vector<OrbitalSpace>& bs, int nobs, int nri)
{ 
    using namespace einsums;
    std::vector<int> o_oei = {0, 0, 1, 
                              0, 1, 1}; 

    timer::push("T and V Matrices");
    int n1, n2, P, Q;
#pragma omp parallel for
    for (int i = 0; i < 3; i++) {
        ( o_oei[i] == 1 ) ? (n1 = nri - nobs, P = nobs) : (n1 = nobs, P = 0);
        ( o_oei[i+3] == 1 ) ? (n2 = nri - nobs, Q = nobs) : (n2 = nobs, Q = 0); 

        timer::push("Get AO and MO Matrices");
        auto t_mo = std::make_shared<Matrix>("MO-based T Integral", n1, n2);
        auto v_mo = std::make_shared<Matrix>("MO-based V Integral", n1, n2);
        {
            auto bs1 = bs[o_oei[i]].basisset();
            auto bs2 = bs[o_oei[i+3]].basisset();
            auto X1 = bs[o_oei[i]].C();
            auto X2 = bs[o_oei[i+3]].C();
            auto t_ao = mints.ao_kinetic(bs1, bs2);
            auto v_ao = mints.ao_potential(bs1, bs2);
            t_mo->transform(X1, t_ao, X2);
            v_mo->transform(X1, v_ao, X2);
            t_ao.reset();
            v_ao.reset();
        }
        timer::pop();

        timer::push("Place MOs in T or V");
        {
            TensorView<double, 2> h_pq{*h, Dim<2>{n1, n2}, Offset<2>{P, Q}};
            for (int p = 0; p < n1; p++) {
                for (int q = 0; q < n2; q++) {
                    h_pq(p,q) = t_mo->get(p,q) + v_mo->get(p,q);
                }
            }
            if ( o_oei[i] != o_oei[i+3] ) {
                TensorView<double, 2> h_qp{*h, Dim<2>{n2, n1}, Offset<2>{Q, P}};
                for (int q = 0; q < n2; q++) {
                    for (int p = 0; p < n1; p++) {
                        h_qp(q,p) = t_mo->get(p,q) + v_mo->get(p,q);
                    }
                }
            } // end of if statement
            t_mo.reset();
            v_mo.reset();
        }
        timer::pop();
    } // end of for loop
    timer::pop(); // T and V Matrices
}

void teints(std::string int_type, einsums::Tensor<double, 4> *ERI, std::vector<OrbitalSpace>& bs, 
		    int nobs, std::shared_ptr<CorrelationFactor> corr)
{
    using namespace einsums; 
    using namespace tensor_algebra;
    using namespace tensor_algebra::index;

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
#pragma omp parallel for collapse(4)
            for (int M = 0; M < bs1->nshell(); M++) {
                for (int N = 0; N < bs3->nshell(); N++) {
                    for (int P = 0; P < bs2->nshell(); P++) {
                        for (int Q = 0; Q < bs4->nshell(); Q++) {
                            ints->compute_shell(M, N, P, Q);

                            for (int m = 0, index = 0; m < bs1->shell(M).nfunction(); m++) {
                                for (int n = 0; n < bs3->shell(N).nfunction(); n++) {
                                    for (int p = 0; p < bs2->shell(P).nfunction(); p++) {
                                        for (int q = 0; q < bs4->shell(Q).nfunction(); q++, index++) {
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
#pragma omp parallel for collapse(2)
            for (int p = 0; p < nbf1; p++) {
                for (int q = 0; q < nmo1; q++) {
                    (*C1)(p,q) = bs[o_tei[i]].C()->get(p,q);	
                }
            }

#pragma omp parallel for collapse(2)
            for (int p = 0; p < nbf2; p++) {
                for (int q = 0; q < nmo2; q++) {
                    (*C2)(p,q) = bs[o_tei[i+1]].C()->get(p,q);	
                }
            }

#pragma omp parallel for collapse(2)
            for (int p = 0; p < nbf3; p++) {
                for (int q = 0; q < nmo3; q++) {
                    (*C3)(p,q) = bs[o_tei[i+2]].C()->get(p,q);	
                }
            }

#pragma omp parallel for collapse(2)
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
#pragma omp parallel for collapse(4)
            for (int i = 0; i < nmo1; i++){
                for (int j = 0; j < nmo2; j++){
                    for (int k = 0; k < nmo3; k++){
                        for (int l = 0; l < nmo4; l++){
                            ERI_PQRS(i,j,k,l) = (*PQRS)(i,j,k,l);
                        }
                    }
                }
            }	
            timer::pop();

            if (nbf4 != nbf1 && nbf4 != nbf2 && nbf4 != nbf3 && int_type != "F2") {
                auto QPSR = std::make_unique<Tensor<double, 4>>("QPSR", nmo2, nmo1, nmo4, nmo3);
                sort(Indices{Q, P, S, R}, &QPSR, Indices{R, S, P, Q}, RSPQ);
                timer::push("Put into ERI Tensor");
                TensorView<double, 4> ERI_QPSR{*ERI, Dim<4>{nmo2, nmo1, nmo4, nmo3}, Offset<4>{J, I, L, K}};
#pragma omp parallel for collapse(4)
                for (int j = 0; j < nmo2; j++){
                    for (int i = 0; i < nmo1; i++){
                        for (int l = 0; l < nmo4; l++){
                            for (int k = 0; k < nmo3; k++){
                                ERI_QPSR(j,i,l,k) = (*QPSR)(j,i,l,k);
                            }
                        }
                    }
                }
                QPSR.reset(nullptr);
                timer::pop();
            } // end of if statement

            if (nbf4 != nbf1 && nbf4 != nbf2 && nbf4 != nbf3 && int_type == "G") {
                auto SRQP = std::make_unique<Tensor<double, 4>>("SRQP", nmo4, nmo3, nmo2, nmo1);
                sort(Indices{S, R, Q, P}, &SRQP, Indices{P, Q, R, S}, PQRS);
                timer::push("Put into ERI Tensor");
                TensorView<double, 4> ERI_SRQP{*ERI, Dim<4>{nmo4, nmo3, nmo2, nmo1}, Offset<4>{L, K, J, I}};
#pragma omp parallel for collapse(4)
                for (int l = 0; l < nmo4; l++){
                    for (int k = 0; k < nmo3; k++){
                        for (int j = 0; j < nmo2; j++){
                            for (int i = 0; i < nmo1; i++){
                                ERI_SRQP(l,k,j,i) = (*SRQP)(l,k,j,i);
                            }
                        }
                    }
                }
                SRQP.reset(nullptr);
                timer::pop();
            } // end of if statement
        }
        RSPQ.reset(nullptr);
        PQRS.reset(nullptr);
        timer::pop(); // Stitch into ERI Tensor
    } // end of for loop
}

void f_mats(einsums::Tensor<double, 2> *f, einsums::Tensor<double, 2> *k, 
            einsums::Tensor<double, 2> *fk, einsums::Tensor<double, 2> *h, 
            einsums::Tensor<double, 4> *G, int nocc, int nri ) 
{
    using namespace einsums;
    using namespace tensor_algebra;
    using namespace tensor_algebra::index;

    timer::push("Forming the f and fk Matrices");

    timer::push("Form J_pq and K_pq");
    (*f) = (*h)(All, All);
    {
        Tensor Id = create_identity_tensor("I", nocc, nocc);
        TensorView<double, 4> G1_view{(*G), Dim<4>{nri, nocc, nri, nocc}, Offset<4>{0,0,0,0}};
        TensorView<double, 4> G2_view{(*G), Dim<4>{nri, nocc, nocc, nri}, Offset<4>{0,0,0,0}};
        auto G1_pqiI = std::make_unique<Tensor<double, 4>>("pqiI", nri, nri, nocc, nocc);
        auto G2_pqiI = std::make_unique<Tensor<double, 4>>("pqiI", nri, nri, nocc, nocc);

        sort(Indices{p, q, i, I}, &G1_pqiI, Indices{p, i, q, I}, G1_view);
        sort(Indices{p, q, i, I}, &G2_pqiI, Indices{p, i, I, q}, G2_view);

        einsum(1.0, Indices{p, q}, &(*f), 2.0, Indices{p, q, i, I}, G1_pqiI, Indices{i, I}, Id);
        einsum(Indices{p, q}, &(*k), Indices{p, q, i, I}, G2_pqiI, Indices{i, I}, Id);
    }
    timer::pop();

    timer::push("Build f and fk Matrices");
    (*fk) = (*f)(All, All);
    tensor_algebra::element([](double const &val1, double const &val2)
                            -> double { return val1 - val2; },
                            &(*f), *k);
    timer::pop();

    timer::pop(); // Forming
}

void V_and_X_mat(einsums::Tensor<double, 4> *VX, einsums::Tensor<double, 4> *F, einsums::Tensor<double, 4> *G_F, 
           einsums::Tensor<double, 4> *FG_F2, int nocc, int nobs, int nri)
{
    using namespace einsums;
    using namespace tensor_algebra;
    using namespace tensor_algebra::index;

    timer::push("Forming the V/X Tensor");
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

    timer::pop(); // Forming
}

void C_mat(einsums::Tensor<double, 4> *C, einsums::Tensor<double, 4> *F, einsums::Tensor<double, 2> *f,
           int nocc, int nobs, int nri)
{
    using namespace einsums;
    using namespace tensor_algebra;
    using namespace tensor_algebra::index;

    timer::push("Forming the C Tensor");
    auto nvir = nobs - nocc;
    Tensor F_oovc = (*F)(Range{0, nocc}, Range{0, nocc}, Range{nocc, nobs}, Range{nobs, nri});
    Tensor f_vc = (*f)(Range{nocc, nobs}, Range{nobs, nri});
    auto tmp = std::make_unique<Tensor<double, 4>>("Temp", nocc, nocc, nvir, nvir);

    einsum(Indices{k, l, a, b}, &tmp, Indices{k, l, a, q}, F_oovc, Indices{b, q}, f_vc);
    sort(Indices{k, l, a, b}, &(*C), Indices{l, k, b, a}, tmp);

    tensor_algebra::element([](double const &val1, double const &val2)
                            -> double { return val1 + val2; }, &(*C), *tmp);

    timer::pop(); // Forming
}

void B_mat(einsums::Tensor<double, 4> *B, einsums::Tensor<double, 4> *Uf, einsums::Tensor<double, 4> *F2,
           einsums::Tensor<double, 4> *F, einsums::Tensor<double, 2> *f, einsums::Tensor<double, 2> *fk, 
           einsums::Tensor<double, 2> *kk, int nocc, int nobs, int ncabs, int nri)
{
    using namespace einsums;
    using namespace tensor_algebra;
    using namespace tensor_algebra::index;
    auto nvir = nobs - nocc;

    printf("\nForming the B Tensor");
    timer::push("Forming the B Tensor");
    timer::push("Allocations");
    auto B_term_2 = std::make_unique<Tensor<double, 4>>("B Term 2", nocc, nocc, nocc, nocc);
    auto B_term_3 = std::make_unique<Tensor<double, 4>>("B Term 3", nocc, nocc, nocc, nocc);
    auto B_term_4 = std::make_unique<Tensor<double, 4>>("B Term 4", nocc, nocc, nocc, nocc);
    auto B_term_5 = std::make_unique<Tensor<double, 4>>("B Term 5", nocc, nocc, nocc, nocc);
    auto B_term_6 = std::make_unique<Tensor<double, 4>>("B Term 6", nocc, nocc, nocc, nocc);
    auto B_term_7 = std::make_unique<Tensor<double, 4>>("B Term 7", nocc, nocc, nocc, nocc);
    auto B_term_8 = std::make_unique<Tensor<double, 4>>("B Term 8", nocc, nocc, nocc, nocc);
    timer::pop();

    timer::push("Term 1");
    (*B) = (*Uf)(Range{0, nocc}, Range{0, nocc}, Range{0, nocc}, Range{0, nocc});
    timer::pop();

    timer::push("Term 2");
    TensorView<double, 4> F2_ooo1{(*F2), Dim<4>{nocc, nocc, nocc, nri}, Offset<4>{0,0,0,0}};
    TensorView<double, 2> fk_o1{(*fk), Dim<2>{nocc, nri}, Offset<2>{0,0}};
    auto tmp_1 = std::make_unique<Tensor<double, 4>>("Temp 1", nocc, nocc, nocc, nocc);

    einsum(Indices{l, k, n, m}, &tmp_1, Indices{l, k, n, I}, F2_ooo1, Indices{m, I}, fk_o1);
    sort(Indices{k, l, m, n}, &B_term_2, Indices{l, k, n, m}, tmp_1);
    tensor_algebra::element([](double const &val1, double const &val2)
                        -> double { return val1 + val2; }, &(*B_term_2), *tmp_1);
    tmp_1.reset();        
    timer::pop();

    timer::push("Term 3");
    TensorView<double, 4> F_oo11_temp{(*F), Dim<4>{nocc, nocc, nri, nri}, Offset<4>{0,0,0,0}};
    Tensor F_oo11 = F_oo11_temp;
    tmp_1 = std::make_unique<Tensor<double, 4>>("Temp 1", nocc, nocc, nri, nri);
    auto tmp_2 = std::make_unique<Tensor<double, 4>>("Temp 2", nocc, nocc, nocc, nocc);

    einsum(Indices{l, k, P, A}, &tmp_1, Indices{l, k, P, C}, F_oo11, Indices{C, A}, *kk);
    einsum(Indices{l, k, n, m}, &tmp_2, Indices{l, k, P, A}, tmp_1, Indices{n, m, P, A}, F_oo11);
    sort(Indices{k, l, m, n}, &B_term_3, Indices{l, k, n, m}, tmp_2);
    tensor_algebra::element([](double const &val1, double const &val2)
                        -> double { return val1 + val2; }, &(*B_term_3), *tmp_2);
    tmp_1.reset();
    tmp_2.reset();
    timer::pop();

    timer::push("Term 4");
    TensorView<double, 4> F_ooo1{(*F), Dim<4>{nocc, nocc, nocc, nri}, Offset<4>{0,0,0,0}};
    tmp_1 = std::make_unique<Tensor<double, 4>>("Temp 1", nocc, nocc, nocc, nri);
    tmp_2 = std::make_unique<Tensor<double, 4>>("Temp 2", nocc, nocc, nocc, nocc);

    einsum(Indices{l, k, j, A}, &tmp_1, Indices{l, k, j, C}, F_ooo1, Indices{C, A}, *f);
    einsum(Indices{l, k, n, m}, &tmp_2, Indices{l, k, j, A}, tmp_1, Indices{n, m, j, A}, F_ooo1);
    sort(Indices{k, l, m, n}, &B_term_4, Indices{l, k, n, m}, tmp_2);
    tensor_algebra::element([](double const &val1, double const &val2)
                    -> double { return val1 + val2; }, &(*B_term_4), *tmp_2);
    tmp_1.reset();
    tmp_2.reset();
    timer::pop();

    timer::push("Term 5");
    TensorView<double, 4> F_ooco{(*F), Dim<4>{nocc, nocc, ncabs, nocc}, Offset<4>{0,0,nobs,0}};
    TensorView<double, 2> f_oo{(*f), Dim<2>{nocc, nocc}, Offset<2>{0,0}};
    tmp_1 = std::make_unique<Tensor<double, 4>>("Temp 1", nocc, nocc, ncabs, nocc);
    tmp_2 = std::make_unique<Tensor<double, 4>>("Temp 2", nocc, nocc, nocc, nocc);

    einsum(Indices{l, k, p, i}, &tmp_1, Indices{l, k, p, j}, F_ooco, Indices{j, i}, f_oo);
    einsum(Indices{l, k, n, m}, &tmp_2, Indices{l, k, p, i}, tmp_1, Indices{n, m, p, i}, F_ooco);
    sort(Indices{k, l, m, n}, &B_term_5, Indices{l, k, n, m}, tmp_2);
    tensor_algebra::element([](double const &val1, double const &val2)
                    -> double { return val1 + val2; }, &(*B_term_5), *tmp_2);
    tmp_1.reset();
    tmp_2.reset();
    timer::pop();

    timer::push("Term 6");
    TensorView<double, 4> F_oovq{(*F), Dim<4>{nocc, nocc, nvir, nobs}, Offset<4>{0,0,nocc,0}};
    TensorView<double, 2> f_pq{(*f), Dim<2>{nobs, nobs}, Offset<2>{0,0}};
    tmp_1 = std::make_unique<Tensor<double, 4>>("Temp 1", nocc, nocc, nvir, nobs);
    tmp_2 = std::make_unique<Tensor<double, 4>>("Temp 2", nocc, nocc, nocc, nocc);

    einsum(Indices{l, k, b, p}, &tmp_1, Indices{l, k, b, r}, F_oovq, Indices{r, p}, f_pq);
    einsum(Indices{l, k, n, m}, &tmp_2, Indices{l, k, b, p}, tmp_1, Indices{n, m, b, p}, F_oovq);
    sort(Indices{k, l, m, n}, &B_term_6, Indices{l, k, n, m}, tmp_2);
    tensor_algebra::element([](double const &val1, double const &val2)
                    -> double { return val1 + val2; }, &(*B_term_6), *tmp_2);
    tmp_1.reset();
    tmp_2.reset();
    timer::pop();

    timer::push("Term 7");
    TensorView<double, 4> F_ooc1{(*F), Dim<4>{nocc, nocc, ncabs, nri}, Offset<4>{0,0,nobs,0}};
    TensorView<double, 2> f_o1{(*f), Dim<2>{nocc, nri}, Offset<2>{0,0}};
    tmp_1 = std::make_unique<Tensor<double, 4>>("Temp 1", nocc, nocc, ncabs, nocc);
    tmp_2 = std::make_unique<Tensor<double, 4>>("Temp 2", nocc, nocc, nocc, nocc);

    einsum(Indices{l, k, p, j}, &tmp_1, Indices{l, k, p, I}, F_ooc1, Indices{j, I}, f_o1);
    einsum(0.0, Indices{l, k, n, m}, &tmp_2, 2.0, Indices{l, k, p, j}, tmp_1, Indices{n, m, p, j}, F_ooco);
    sort(Indices{k, l, m, n}, &B_term_7, Indices{l, k, n, m}, tmp_2);
    tensor_algebra::element([](double const &val1, double const &val2)
                    -> double { return val1 + val2; }, &(*B_term_7), *tmp_2);
    tmp_1.reset();
    tmp_2.reset();
    timer::pop();

    timer::push("Term 8");
    TensorView<double, 4> F_oovc{(*F), Dim<4>{nocc, nocc, nvir, ncabs}, Offset<4>{0,0,nocc,nobs}};
    TensorView<double, 2> f_pc{(*f), Dim<2>{nobs, ncabs}, Offset<2>{0,nobs}};
    tmp_1 = std::make_unique<Tensor<double, 4>>("Temp 1", nocc, nocc, nvir, ncabs);
    tmp_2 = std::make_unique<Tensor<double, 4>>("Temp 2", nocc, nocc, nocc, nocc);

    einsum(Indices{l, k, b, q}, &tmp_1, Indices{l, k, b, r}, F_oovq, Indices{r, q}, f_pc);
    einsum(0.0, Indices{l, k, n, m}, &tmp_2, 2.0, Indices{l, k, b, q}, tmp_1, Indices{n, m, b, q}, F_oovc);
    sort(Indices{k, l, m, n}, &B_term_8, Indices{l, k, n, m}, tmp_2);
    tensor_algebra::element([](double const &val1, double const &val2)
                    -> double { return val1 + val2; }, &(*B_term_8), *tmp_2);
    tmp_1.reset();
    tmp_2.reset();
    timer::pop();

    timer::push("Building the B Matrix");
    tmp_1 = std::make_unique<Tensor<double, 4>>("Temp 1", nocc, nocc, nocc, nocc);
    tensor_algebra::element([](double const &val1, double const &val2, double const &val3,
                               double const &val4, double const &val5, double const &val6,
                               double const &val7, double const &val8) 
                            -> double { return val1 + val2 - val3 - val4 + val5 - val6
                                        - val7 - val8; }, 
                            &(*B), *B_term_2, *B_term_3, *B_term_4, *B_term_5,
                            *B_term_6, *B_term_7, *B_term_8);
    sort(Indices{m, n, k, l}, &tmp_1, Indices{k, l, m, n}, (*B));
    tensor_algebra::element([](double const &val1, double const &val2)
                    -> double { return 0.5 * (val1 + val2); }, &(*B), *tmp_1);
    timer::pop();
    timer::pop(); // Forming
}

void D_mat(einsums::Tensor<double, 4> *D, einsums::Tensor<double, 2> *f, int nocc, int nobs)
{
    using namespace einsums;
    using namespace tensor_algebra;

    printf("\nForming the D Tensor\n");

    timer::push("Forming the D Tensor");
    timer::push("Buliding D Tensor");
#pragma omp parallel for
    for (int i = 0; i < nocc; i++) {
        for (int j = 0; j < nocc; j++) {
            for (int a = nocc; a < nobs; a++) {
                for (int b = nocc; b < nobs; b++) {
                    auto denom = (*f)(a,a) + (*f)(b,b) - (*f)(i,i) - (*f)(j,j);
                    (*D)(i,j,a-nocc,b-nocc) = (1 / denom);
                }
            }
        }
    }
    timer::pop();
    timer::pop(); // Forming
}

double t_(int p, int q, int r, int s) 
{
    auto t_amp = 0.0;
    if (p == r && q == s && p != q) {
        t_amp = 3.0/8.0;
    } else if (q == r && p == s && p != q) {
        t_amp = 1.0/8.0;
    } else if (p == q && p == r && p == s) {
        t_amp = 0.5;
    }
    return t_amp;
}

std::pair<double, double> V_Tilde(einsums::TensorView<double, 2> V_, einsums::Tensor<double, 4> *C, 
                                  einsums::TensorView<double, 2> K_ij, einsums::TensorView<double, 2> D_ij,
                                  int i, int j, int nocc, int nobs)
{
    using namespace einsums;
    using namespace tensor_algebra;
    using namespace tensor_algebra::index;

    auto nvir = nobs - nocc;
    auto V_s = 0.0;
    auto V_t = 0.0;
    int kd;

    timer::push("Forming the V_Tilde Matrices");
    timer::push("Allocations");
    auto V_ij = std::make_unique<Tensor<double, 2>>("V(i, j, :, :)", nocc, nocc);
    auto KD = std::make_unique<Tensor<double, 2>>("Temp 1", nvir, nvir);
    (*V_ij) = V_;
    timer::pop(); 

    timer::push("Perform einsums");
    einsum(Indices{a, b}, &KD, Indices{a, b}, K_ij, Indices{a, b}, D_ij);
    einsum(1.0, Indices{k, l}, &V_ij, -1.0, Indices{k, l, a, b}, *C, Indices{a, b}, KD);
    ( i == j ) ? ( kd = 1 ) : ( kd = 2 );
    timer::pop();

    timer::push("V Singlet Vector");
    V_s += 0.5 * (t_(i,j,i,j) + t_(i,j,j,i)) * kd * ((*V_ij)(i,j) + (*V_ij)(j,i));
    timer::pop();

    if ( i != j ) {
        timer::push("V Triplet Vector");
        V_t += 0.5 * (t_(i,j,i,j) - t_(i,j,j,i)) * kd * ((*V_ij)(i,j) - (*V_ij)(j,i));
        timer::pop();
    }
    timer::pop(); // Forming
    return {V_s, V_t};
}

std::pair<double, double> B_Tilde(einsums::Tensor<double, 4> *B_, einsums::Tensor<double, 4> *C, 
                                  einsums::TensorView<double, 2> D_ij, 
                                  int i, int j, int nocc, int nobs)
{
    using namespace einsums;
    using namespace tensor_algebra;
    using namespace tensor_algebra::index;

    auto nvir = nobs - nocc;
    auto B_s = 0.0;
    auto B_t = 0.0;
    int kd;

    timer::push("Forming the B_Tilde Matrices");
    timer::push("Allocations");
    auto B_ij = std::make_unique<Tensor<double, 4>>("B = B - X * (fii +fjj)", nocc, nocc, nocc, nocc);
    auto CD = std::make_unique<Tensor<double, 4>>("Temp 1", nocc, nocc, nvir, nvir);
    (*B_ij) = (*B_);
    ( i == j ) ? ( kd = 1 ) : ( kd = 2 );
    timer::pop(); 

    timer::push("Perform einsums");
    einsum(Indices{k, l, a, b}, &CD, Indices{k, l, a, b}, *C, Indices{a, b}, D_ij);
    einsum(1.0, Indices{k, l, m, n}, &B_ij, -1.0, Indices{m, n, a, b}, *C,
                                                  Indices{k, l, a, b}, CD);
    timer::pop();

    timer::push("B Singlet Matrix");
    B_s += 0.125 * (t_(i,j,i,j) + t_(i,j,j,i)) * kd 
               * ((*B_ij)(i,j,i,j) + (*B_ij)(j,i,i,j))
               * (t_(i,j,i,j) + t_(i,j,j,i)) * kd;
    timer::pop();

    if ( i != j ) {
        timer::push("B Triplet Matrix");
        B_t += 0.125 * (t_(i,j,i,j) - t_(i,j,j,i)) * kd
                   * ((*B_ij)(i,j,i,j) - (*B_ij)(j,i,i,j))
                   * (t_(i,j,i,j) - t_(i,j,j,i)) * kd;
        timer::pop();
    }
    timer::pop(); // Forming
    return {B_s, B_t};
}

extern "C" PSI_API
SharedWavefunction MP2F12(SharedWavefunction ref_wfn, Options& options)
{
    int PRINT = options.get_int("PRINT");
    bool WRITE_INTS = options.get_bool("WRITE_INTS");
    bool READ_INTS = options.get_bool("READ_INTS");
    auto FRZN = options.get_str("FREEZE_CORE");

    outfile->Printf("   --------------------------------------------\n");
    outfile->Printf("                    MP2-F12/3C                 \n");
    outfile->Printf("   --------------------------------------------\n\n");

    // Get the AO basis sets, OBS and CABS //
    outfile->Printf("  ==> Forming the OBS and CABS <==\n");
    OrbitalSpace OBS = ref_wfn->alpha_orbital_space("p","SO","ALL");
    OrbitalSpace RI = OrbitalSpace::build_ri_space(ref_wfn->get_basisset("CABS"), 1.0e-8);
    OrbitalSpace CABS = OrbitalSpace::build_cabs_space(OBS, RI, 1.0e-6);
    std::vector<OrbitalSpace> bs = {OBS, CABS};
    auto nobs = OBS.dim().max();
    auto nri = CABS.dim().max() + OBS.dim().max();
    auto nocc = ref_wfn->doccpi()[0];
    auto ncabs = nri - nobs;
    auto nfrzn = 0;

    outfile->Printf("  F12 MO Spaces: \n");
    outfile->Printf("     NOCC: %3d \n", nocc);
    outfile->Printf("     NVIR: %3d \n", nobs-nocc);
    outfile->Printf("     NOBS: %3d \n", nobs);
    outfile->Printf("    NCABS: %3d \n", ncabs);

    if (FRZN == "TRUE") {
        Dimension dfrzn = ref_wfn->frzcpi();
        dfrzn.print();
        nfrzn = dfrzn.max();
    }

    // Form the one-electron integrals //
    outfile->Printf("\n  ==> Forming the Integrals <==\n");
    using namespace einsums; 

    if (READ_INTS) {
        // Disable HDF5 diagnostic reporting.
        H5Eset_auto(0, nullptr, nullptr);
        state::data = h5::open("Data.h5", H5F_ACC_RDWR);
    }

    if (WRITE_INTS) {
        // Disable HDF5 diagnostic reporting.
        H5Eset_auto(0, nullptr, nullptr);
        state::data = h5::create("Data.h5", H5F_ACC_TRUNC);
    }

    timer::initialize();
    timer::push("Form all the INTS");
    
    timer::push("OEINTS");
    timer::push("OEINTS Allocations");
    auto h = std::make_unique<Tensor<double, 2>>("MO One-Electron Integrals", nri, nri);
    timer::pop();

    if (READ_INTS) {
        read_ints_disk(h.get(), nri, nri);
    } else if (WRITE_INTS) {
        write_ints_disk(h.get(), nri, nri);
    } else {
        MintsHelper mints(MintsHelper(OBS.basisset(), options, PRINT));
        outfile->Printf("      One-Electron Integrals\n");
        oeints(mints, h.get(), bs, nobs, nri);
    }
    timer::pop(); // OEINTS

    // Form the two-electron integrals //
    timer::push("TEINTS");
    std::vector<std::string> teint = {"FG","Uf","G","F","F2"};
    std::shared_ptr<CorrelationFactor> cgtg(new FittedSlaterCorrelationFactor(1.0));

    timer::push("TEINTS Allocations");
    auto G = std::make_unique<Tensor<double, 4>>("MO G Tensor", nri, nobs, nri, nri);
    auto F = std::make_unique<Tensor<double, 4>>("MO F12 Tensor", nobs, nobs, nri, nri);
    auto F2 = std::make_unique<Tensor<double, 4>>("MO F12_Squared Tensor", nobs, nobs, nobs, nri);
    auto FG = std::make_unique<Tensor<double, 4>>("MO F12G12 Tensor", nobs, nobs, nobs, nobs);
    auto Uf = std::make_unique<Tensor<double, 4>>("MO F12_DoubleCommutator Tensor", nobs, nobs, nobs, nobs);
    timer::pop();

    if (READ_INTS){
        read_ints_disk(G.get(), nri, nobs, nri, nri);
        read_ints_disk(F.get(), nobs, nobs, nri, nri);
        read_ints_disk(F2.get(), nobs, nobs, nobs, nri);
        read_ints_disk(FG.get(), nobs, nobs, nobs, nobs);
        read_ints_disk(Uf.get(), nobs, nobs, nobs, nobs);
    } else if (WRITE_INTS) {
        write_ints_disk(G.get(), nri, nobs, nri, nri);
        write_ints_disk(F.get(), nobs, nobs, nri, nri);
        write_ints_disk(F2.get(), nobs, nobs, nobs, nri);
        write_ints_disk(FG.get(), nobs, nobs, nobs, nobs);
        write_ints_disk(Uf.get(), nobs, nobs, nobs, nobs);
    } else {
        for (int i = 0; i < teint.size(); i++){
            if ( teint[i] == "F" ){
                outfile->Printf("      F Integral\n");
                teints(teint[i], F.get(), bs, nobs, cgtg);
            } else if ( teint[i] == "FG" ){
                outfile->Printf("      FG Integral\n");
                teints(teint[i], FG.get(), bs, nobs, cgtg);
            } else if ( teint[i] == "F2" ){
                outfile->Printf("      F Squared Integral\n");
                teints(teint[i], F2.get(), bs, nobs, cgtg);
            } else if ( teint[i] == "Uf" ){
                outfile->Printf("      F Double Commutator Integral\n");
                teints(teint[i], Uf.get(), bs, nobs, cgtg);
            } else {
                outfile->Printf("      G Integral\n");
                teints(teint[i], G.get(), bs, nobs, cgtg);
            }
        }
    }
    timer::pop(); // TEINTS

    // Form the F12 Matrices //
    outfile->Printf("\n  ==> Forming the F12 Intermediate Tensors <==\n");
    timer::push("F12 INTS");
    timer::push("F12 INTS Allocations");
    auto f = std::make_unique<Tensor<double, 2>>("Fock Matrix", nri, nri);
    auto k = std::make_unique<Tensor<double, 2>>("Exchange MO Integral", nri, nri);
    auto fk = std::make_unique<Tensor<double, 2>>("Fock-Exchange Matrix", nri, nri);
    auto V = std::make_unique<Tensor<double, 4>>("V Intermediate Tensor", nocc, nocc, nocc, nocc);
    auto X = std::make_unique<Tensor<double, 4>>("X Intermediate Tensor", nocc, nocc, nocc, nocc);
    auto C = std::make_unique<Tensor<double, 4>>("C Intermediate Tensor", nocc, nocc, nobs-nocc, nobs-nocc);
    auto B = std::make_unique<Tensor<double, 4>>("B Intermediate Tensor", nocc, nocc, nocc, nocc);
    timer::pop();

    if (READ_INTS) {
        read_ints_disk(f.get(), nri, nri);
        read_ints_disk(k.get(), nri, nri);
        read_ints_disk(fk.get(), nri, nri);
        read_ints_disk(V.get(), nocc, nocc, nocc, nocc);
        read_ints_disk(X.get(), nocc, nocc, nocc, nocc);
        read_ints_disk(C.get(), nocc, nocc, nobs-nocc, nobs-nocc);
        read_ints_disk(B.get(), nocc, nocc, nocc, nocc);
    } else if (WRITE_INTS) {
        write_ints_disk(f.get(), nri, nri);
        write_ints_disk(k.get(), nri, nri);
        write_ints_disk(fk.get(), nri, nri);
        write_ints_disk(V.get(), nocc, nocc, nocc, nocc);
        write_ints_disk(X.get(), nocc, nocc, nocc, nocc);
        write_ints_disk(C.get(), nocc, nocc, nobs-nocc, nobs-nocc);
        write_ints_disk(B.get(), nocc, nocc, nocc, nocc);
    } else {
        outfile->Printf("      Fock Matrix\n");
        f_mats(f.get(), k.get(), fk.get(), h.get(), G.get(), nocc, nri);
        outfile->Printf("      V Intermediate\n");
        V_and_X_mat(V.get(), F.get(), G.get(), FG.get(), nocc, nobs, nri);
        outfile->Printf("      X Intermediate\n");
        V_and_X_mat(X.get(), F.get(), F.get(), F2.get(), nocc, nobs, nri);
        outfile->Printf("      C Intermediate\n");
        C_mat(C.get(), F.get(), f.get(), nocc, nobs, nri);
        outfile->Printf("      B Intermediate\n");
        B_mat(B.get(), Uf.get(), F2.get(), F.get(), f.get(), fk.get(), k.get(), nocc, nobs, ncabs, nri);
    }
    timer::pop(); // F12 INTS

    timer::pop(); // Form all the INTS

    // Compute the MP2F12/3C Energy //
    outfile->Printf("\n  ==> Computing F12/3C Energy Correction <==\n");
    timer::push("MP2F12/3C Energy");
    timer::push("Allocations");
    auto E_f12_s = 0.0;
    auto E_f12_t = 0.0;
    auto E_core = 0.0;
    auto D = std::make_unique<Tensor<double, 4>>("D Tensor", nocc, nocc, nobs-nocc, nobs-nocc);
    auto G_ = (*G)(Range{0, nocc}, Range{0, nocc}, Range{nocc, nobs}, Range{nocc, nobs});
    int kd;
    timer::pop();

    D_mat(D.get(), f.get(), nocc, nobs);

    outfile->Printf("  \n");
    outfile->Printf("  %1s   %1s  |     %14s     %14s     %12s \n",
                    "i", "j", "E_F12(Singlet)", "E_F12(Triplet)", "E_F12");
    outfile->Printf(" ----------------------------------------------------------------------\n");
    for (int i = nfrzn; i < nocc; i++) {
        for (int j = i; j < nocc; j++) {
            // Allocations
            auto X_ = std::make_unique<Tensor<double, 4>>("Scaled X", nocc, nocc, nocc, nocc);
            auto B_ = std::make_unique<Tensor<double, 4>>("B ij", nocc, nocc, nocc, nocc);
            (*X_) = (*X)(Range{0, nocc}, Range{0, nocc}, Range{0, nocc}, Range{0, nocc});
            (*B_) = (*B)(Range{0, nocc}, Range{0, nocc}, Range{0, nocc}, Range{0, nocc});
            // Building B_
            auto f_scale = (*f)(i,i) + (*f)(j,j);
            linear_algebra::scale(f_scale, &(*X_));
            tensor_algebra::element([](double const &Bval, double const &Xval) 
                                    -> double { return Bval - Xval; }, B_.get(), *X_);
            // Getting V_Tilde and B_Tilde
            auto V_ = TensorView<double, 2>{(*V), Dim<2>{nocc, nocc}, Offset<4>{i, j, 0, 0}, 
                                            Stride<2>{(*V).stride(2), (*V).stride(3)}};
            auto K_ = TensorView<double, 2>{G_, Dim<2>{nobs-nocc, nobs-nocc}, Offset<4>{i, j, 0, 0},
                                            Stride<2>{G_.stride(2), G_.stride(3)}};
            auto D_ = TensorView<double, 2>{(*D), Dim<2>{nobs-nocc, nobs-nocc}, Offset<4>{i, j, 0, 0},
                                            Stride<2>{(*D).stride(2), (*D).stride(3)}};
            auto VT = V_Tilde(V_, C.get(), K_, D_, i, j, nocc, nobs);
            auto BT = B_Tilde(B_.get(), C.get(), D_, i, j, nocc, nobs);
            // Computing the energy
            ( i == j ) ? ( kd = 1 ) : ( kd = 2 );
            auto E_s = kd * (VT.first + BT.first);
            E_f12_s = E_f12_s + E_s;
            auto E_t = 0.0;
            if ( i != j ) {
                E_t = kd * (VT.second + BT.second);
                E_f12_t = E_f12_t + E_t;
            }
            auto E_f = E_s + (3.0 * E_t);
            outfile->Printf("%3d %3d  |   %16.12f   %16.12f     %16.12f \n", i+1, j+1, E_s, E_t, E_f);
            X_.reset();
            B_.reset();
        }
    }
    E_f12_t = 3.0 * E_f12_t;
    auto E_f12 = E_f12_s + E_f12_t; 

    auto e_mp2 = ref_wfn->energy();
    outfile->Printf("  \n");
    outfile->Printf(" Total MP2-F12/3C Energy:             %16.12f \n", e_mp2 + E_f12);
    outfile->Printf("    MP2 Energy:                       %16.12f \n", e_mp2);
    outfile->Printf("    F12/3C Singlet Correction:        %16.12f \n", E_f12_s);
    outfile->Printf("    F12/3C Triplet Correction:        %16.12f \n", E_f12_t);
    outfile->Printf("    F12/3C Correction:                %16.12f \n", E_f12);
    timer::pop(); // MP2F12/3C Energy

    timer::report();
    timer::finalize();

    // Typically you would build a new wavefunction and populate it with data
    return ref_wfn;
}

}} // End namespaces
