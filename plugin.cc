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
#include "f12_intermediates.h"

#include "psi4/libpsi4util/PsiOutStream.h"

#include "psi4/libmints/basisset.h"
#include "psi4/libmints/dimension.h"
#include "psi4/libmints/integralparameters.h"
#include "psi4/libmints/matrix.h"
#include "psi4/libmints/mintshelper.h"
#include "psi4/libmints/orbitalspace.h"
#include "psi4/libmints/wavefunction.h"

#include "einsums/LinearAlgebra.hpp"
#include "einsums/Sort.hpp"
#include "einsums/Tensor.hpp"
#include "einsums/TensorAlgebra.hpp"

namespace psi{ namespace mp2f12 {

extern "C" PSI_API
int read_options(std::string name, Options& options)
{
    if (name == "MP2F12"|| options.read_globals()) {
        /*- The amount of information printed to the output file -*/
        options.add_int("PRINT", 1);
        /*- Whether to compute the CABS Singles Correction -*/
        options.add_bool("CABS_SINGLES", true);
        /*- Choose conventional or density-fitted. Default to CONV -*/
        options.add_str("F12_TYPE", "CONV");
    }

    return true;
}

void oeints(MintsHelper& mints, einsums::Tensor<double, 2> *h,
		    std::vector<OrbitalSpace>& bs, const int& nobs, const int& nri)
{ 
    using namespace einsums;

    int nthreads = 1;
#ifdef _OPENMP
    nthreads = Process::environment.get_n_threads();
#endif

    std::vector<int> o_oei = {0, 0, 1, 
                              0, 1, 1}; 

    int n1, n2, P, Q;
    for (int i = 0; i < 3; i++) {
        ( o_oei[i] == 1 ) ? (n1 = nri - nobs, P = nobs) : (n1 = nobs, P = 0);
        ( o_oei[i+3] == 1 ) ? (n2 = nri - nobs, Q = nobs) : (n2 = nobs, Q = 0); 

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

        {
            TensorView<double, 2> h_pq{*h, Dim<2>{n1, n2}, Offset<2>{P, Q}};
	        #pragma omp parallel for collapse(2) num_threads(nthreads)
            for (int p = 0; p < n1; p++) {
                for (int q = 0; q < n2; q++) {
                    h_pq(p,q) = t_mo->get(p,q) + v_mo->get(p,q);
                }
            }
            if ( o_oei[i] != o_oei[i+3] ) {
                TensorView<double, 2> h_qp{*h, Dim<2>{n2, n1}, Offset<2>{Q, P}};
		        #pragma omp parallel for collapse(2) num_threads(nthreads)
                for (int q = 0; q < n2; q++) {
                    for (int p = 0; p < n1; p++) {
                        h_qp(q,p) = t_mo->get(p,q) + v_mo->get(p,q);
                    }
                }
            } // end of if statement
            t_mo.reset();
            v_mo.reset();
        }
    } // end of for loop
}

void D_mat(einsums::Tensor<double, 4> *D, einsums::Tensor<double, 2> *f, const int& nocc, const int& nobs)
{
    using namespace einsums;
    using namespace tensor_algebra;

    int nthreads = 1;
#ifdef _OPENMP
    nthreads = Process::environment.get_n_threads();
#endif

    #pragma omp parallel for collapse(4) num_threads(nthreads)
    for (int i = 0; i < nocc; i++) {
        for (int j = 0; j < nocc; j++) {
            for (int a = nocc; a < nobs; a++) {
                for (int b = nocc; b < nobs; b++) {
                    auto denom = (*f)(a,a) + (*f)(b,b) - (*f)(i,i) - (*f)(j,j);
                    (*D)(i, j, a-nocc, b-nocc) = (1 / denom);
                }
            }
        }
    }
}

double t_(const int& p, const int& q, const int& r, const int& s)
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

std::pair<double, double> V_Tilde(einsums::TensorView<double, 2>& V_, einsums::Tensor<double, 4> *C,
                                  einsums::TensorView<double, 2>& K_ij, einsums::TensorView<double, 2>& D_ij,
                                  const int& i, const int& j, const int& nocc, const int& nobs)
{
    using namespace einsums;
    using namespace tensor_algebra;
    using namespace tensor_algebra::index;

    auto nvir = nobs - nocc;
    auto V_s = 0.0;
    auto V_t = 0.0;
    int kd;

    auto V_ij = std::make_unique<Tensor<double, 2>>("V(i, j, :, :)", nocc, nocc);
    auto KD = std::make_unique<Tensor<double, 2>>("Temp 1", nvir, nvir);
    (*V_ij) = V_;

    einsum(Indices{a, b}, &KD, Indices{a, b}, K_ij, Indices{a, b}, D_ij);
    einsum(1.0, Indices{k, l}, &V_ij, -1.0, Indices{k, l, a, b}, *C, Indices{a, b}, KD);
    ( i == j ) ? ( kd = 1 ) : ( kd = 2 );

    V_s += 0.5 * (t_(i,j,i,j) + t_(i,j,j,i)) * kd * ((*V_ij)(i,j) + (*V_ij)(j,i));

    if ( i != j ) {
        V_t += 0.5 * (t_(i,j,i,j) - t_(i,j,j,i)) * kd * ((*V_ij)(i,j) - (*V_ij)(j,i));
    }
    return {V_s, V_t};
}

std::pair<double, double> B_Tilde(einsums::Tensor<double, 4>& B, einsums::Tensor<double, 4> *C, 
                                  einsums::TensorView<double, 2>& D_ij, 
                                  const int& i, const int& j, const int& nocc, const int& nobs)
{
    using namespace einsums;
    using namespace tensor_algebra;
    using namespace tensor_algebra::index;

    auto nvir = nobs - nocc;
    auto B_s = 0.0;
    auto B_t = 0.0;
    int kd;

    Tensor<double, 4> B_ij{"B = B - X * (fii +fjj)", nocc, nocc, nocc, nocc};
    Tensor<double, 4> CD{"Temp 1", nocc, nocc, nvir, nvir};
    B_ij = B;
    ( i == j ) ? ( kd = 1 ) : ( kd = 2 );

    einsum(Indices{k, l, a, b}, &CD, Indices{k, l, a, b}, *C, Indices{a, b}, D_ij);
    einsum(1.0, Indices{k, l, m, n}, &B_ij, -1.0, Indices{m, n, a, b}, *C,
                                                  Indices{k, l, a, b}, CD);

    B_s += 0.125 * (t_(i,j,i,j) + t_(i,j,j,i)) * kd 
                 * (B_ij(i,j,i,j) + B_ij(j,i,i,j))
                 * (t_(i,j,i,j) + t_(i,j,j,i)) * kd;

    if ( i != j ) {
        B_t += 0.125 * (t_(i,j,i,j) - t_(i,j,j,i)) * kd
                     * (B_ij(i,j,i,j) - B_ij(j,i,i,j))
                     * (t_(i,j,i,j) - t_(i,j,j,i)) * kd;
    }
    return {B_s, B_t};
}

double compute_energy(einsums::Tensor<double,4> *G, einsums::Tensor<double,4> *X, 
                      einsums::Tensor<double,4> *B, einsums::Tensor<double,4> *V,
                      einsums::Tensor<double,2> *f, einsums::Tensor<double,4> *C, 
                      einsums::Tensor<double,4> *D, const int& nocc, const int& nobs, const int& nfrzn)
{
    using namespace einsums;
    using namespace tensor_algebra;
    using namespace tensor_algebra::index;

    int nthreads = 1;
#ifdef _OPENMP
    nthreads = Process::environment.get_n_threads();
#endif

    auto E_f12_s = 0.0;
    auto E_f12_t = 0.0;
    auto G_ = (*G)(Range{0, nocc}, Range{0, nocc}, Range{nocc, nobs}, Range{nocc, nobs});
    int kd;

    outfile->Printf("  \n");
    outfile->Printf("  %1s   %1s  |     %14s     %14s     %12s \n",
                    "i", "j", "E_F12(Singlet)", "E_F12(Triplet)", "E_F12");
    outfile->Printf(" ----------------------------------------------------------------------\n");
    #pragma omp parallel for ordered num_threads(nthreads) reduction(+:E_f12_s,E_f12_t)
    for (int i = nfrzn; i < nocc; i++) {
        #pragma omp ordered
        for (int j = i; j < nocc; j++) {
            // Allocations
            Tensor<double, 4> X_{"Scaled X", nocc, nocc, nocc, nocc};
            Tensor<double, 4> B_{"B ij", nocc, nocc, nocc, nocc};
            X_ = (*X)(Range{0, nocc}, Range{0, nocc}, Range{0, nocc}, Range{0, nocc});
            B_ = (*B)(Range{0, nocc}, Range{0, nocc}, Range{0, nocc}, Range{0, nocc});
            // Building B_
            auto f_scale = (*f)(i,i) + (*f)(j,j);
            linear_algebra::scale(f_scale, &X_);
            tensor_algebra::element([](double const &Bval, double const &Xval)
                                    -> double { return Bval - Xval; }, &B_, X_);
            // Getting V_Tilde and B_Tilde
            auto V_ = TensorView<double, 2>{(*V), Dim<2>{nocc, nocc}, Offset<4>{i, j, 0, 0},
                                            Stride<2>{(*V).stride(2), (*V).stride(3)}};
            auto K_ = TensorView<double, 2>{G_, Dim<2>{nobs-nocc, nobs-nocc}, Offset<4>{i, j, 0, 0},
                                            Stride<2>{G_.stride(2), G_.stride(3)}};
            auto D_ = TensorView<double, 2>{(*D), Dim<2>{nobs-nocc, nobs-nocc}, Offset<4>{i, j, 0, 0},
                                            Stride<2>{(*D).stride(2), (*D).stride(3)}};
            auto VT = V_Tilde(V_, C, K_, D_, i, j, nocc, nobs);
            auto BT = B_Tilde(B_, C, D_, i, j, nocc, nobs);
            // Computing the energy
            ( i == j ) ? ( kd = 1 ) : ( kd = 2 );
            auto E_s = kd * (VT.first + BT.first);
            E_f12_s += E_s;
            auto E_t = 0.0;
            if ( i != j ) {
                E_t = 3.0 * kd * (VT.second + BT.second);
                E_f12_t += E_t;
            }
            auto E_f = E_s + E_t;
            outfile->Printf("%3d %3d  |   %16.12f   %16.12f     %16.12f \n", i+1, j+1, E_s, E_t, E_f);
        }
    }
    outfile->Printf("\n  F12/3C Singlet Correlation:      %16.12f \n", E_f12_s);
    outfile->Printf("  F12/3C Triplet Correlation:      %16.12f \n", E_f12_t);

    auto E_f12 = E_f12_s + E_f12_t;

    return E_f12;
}

double cabs_singles(einsums::Tensor<double,2> *f, const int& nocc, const int& nri)
{
    using namespace einsums;
    using namespace linear_algebra;

    int nthreads = 1;
#ifdef _OPENMP
    nthreads = Process::environment.get_n_threads();
#endif

    int all_vir = nri - nocc;

    // Diagonalize f_ij and f_AB
    Tensor<double, 2> C_ij{"occupied e-vecs", nocc, nocc};
    Tensor<double, 2> C_AB{"vir and CABS e-vecs", all_vir, all_vir};

    Tensor<double, 1> e_ij{"occupied e-vals", nocc};
    Tensor<double, 1> e_AB{"vir and CABS e-vals", all_vir};
    {
        C_ij = (*f)(Range{0, nocc}, Range{0, nocc});
        C_AB = (*f)(Range{nocc, nri}, Range{nocc, nri});

        syev(&C_ij, &e_ij);
        syev(&C_AB, &e_AB);
    }

    // Form f_iA
    Tensor<double, 2> f_iA{"Fock Occ-All_vir", nocc, all_vir};
    {
        Tensor f_view = (*f)(Range{0,nocc}, Range{nocc, nri});

        gemm<false, false>(1.0, C_ij, gemm<false, true>(1.0, f_view, C_AB), 0.0, &(f_iA));
    }

    double singles = 0;
    #pragma omp parallel for collapse(2) num_threads(nthreads) reduction(+:singles)
    for (int A = 0; A < all_vir; A++) {
        for (int i = 0; i < nocc; i++) {
            singles += 2 * pow(f_iA(i, A), 2) / (e_ij(i) - e_AB(A));
        }
    }

    return singles;
}

void print_header(std::string f12_type)
{
        int nthreads = 1;
#ifdef _OPENMP
    nthreads = Process::environment.get_n_threads();
#endif

    outfile->Printf(" -----------------------------------------------------------\n");
    if (f12_type == "DF") {
        outfile->Printf("                      DF-MP2-F12/3C(FIX)                    \n");
        outfile->Printf("       2nd Order Density-Fitted Explicitly Correlated       \n");
        outfile->Printf("                    Moeller-Plesset Theory                  \n");
        outfile->Printf("                RMP2 Wavefunction, %2d Threads              \n\n", nthreads);
        outfile->Printf("               Erica Mitchell and Justin Turney             \n");
    } else {
        outfile->Printf("                       MP2-F12/3C(FIX)                      \n");
        outfile->Printf("   2nd Order Explicitly Correlated Moeller-Plesset Theory   \n");
        outfile->Printf("               RMP2 Wavefunction, %2d Threads               \n\n", nthreads);
        outfile->Printf("              Erica Mitchell and Justin Turney              \n");
    }
    outfile->Printf(" -----------------------------------------------------------\n\n");
}

extern "C" PSI_API
SharedWavefunction MP2F12(SharedWavefunction ref_wfn, Options& options)
{
    int PRINT = options.get_int("PRINT");
    auto FRZN = options.get_str("FREEZE_CORE");
    bool SINGLES = options.get_bool("CABS_SINGLES");
    auto F12_TYPE = options.get_str("F12_TYPE");

    using namespace einsums;

    print_header(F12_TYPE);

    /* Get the AO basis sets, OBS and CABS */
    outfile->Printf(" ===> Forming the OBS and CABS <===\n\n");

    outfile->Printf("  Orbital Basis Set (OBS)\n");
    OrbitalSpace OBS = ref_wfn->alpha_orbital_space("p","SO","ALL");
    OBS.basisset()->print();

    outfile->Printf("  Complimentary Auxiliary Basis Set (CABS)\n");
    OrbitalSpace RI = OrbitalSpace::build_ri_space(ref_wfn->get_basisset("CABS"), 1.0e-8);
    OrbitalSpace CABS = OrbitalSpace::build_cabs_space(OBS, RI, 1.0e-6);
    CABS.basisset()->print();
    std::vector<OrbitalSpace> bs = {OBS, CABS};
    
    std::shared_ptr<BasisSet> DFBS;
    if (F12_TYPE == "DF") {
        outfile->Printf("  Auxiliary Basis Set\n");
        DFBS = ref_wfn->get_basisset("DF_BASIS_MP2");
        DFBS->print();
    }

    auto nobs = OBS.dim().max();
    auto nri = CABS.dim().max() + OBS.dim().max();
    auto nocc = ref_wfn->doccpi()[0];
    auto ncabs = nri - nobs;
    auto nfrzn = 0;
    auto naux = 0;

    if (F12_TYPE == "DF") {
        naux = DFBS->nbf();
    }

    outfile->Printf("  ----------------------------------------\n");
    outfile->Printf("     %5s  %5s   %5s  %5s  %5s   \n", "NOCC", "NOBS", "NCABS", "NRI", "NAUX");
    outfile->Printf("  ----------------------------------------\n");
    outfile->Printf("     %5d  %5d   %5d  %5d  %5d   \n", nocc, nobs, ncabs, nri, naux);
    outfile->Printf("  ----------------------------------------\n");

    if (FRZN == "TRUE") {
        Dimension dfrzn = ref_wfn->frzcpi();
        outfile->Printf("\n");
        dfrzn.print();
        nfrzn = dfrzn.max();
    }

    /* Form the one-electron integrals */
    outfile->Printf("\n ===> Forming the Integrals <===\n");

    auto h = std::make_unique<Tensor<double, 2>>("MO One-Electron Integrals", nri, nri);

    MintsHelper mints(MintsHelper(OBS.basisset(), options, PRINT));
    outfile->Printf("   One-Electron Integrals\n");
    oeints(mints, h.get(), bs, nobs, nri);

    /* Form the two-electron integrals */
    std::vector<std::string> teint = {"FG","Uf","G","F","F2"};
    std::shared_ptr<CorrelationFactor> cgtg(new FittedSlaterCorrelationFactor(1.0));

    auto G = std::make_unique<Tensor<double, 4>>("MO G Tensor", nri, nobs, nri, nri);
    auto F = std::make_unique<Tensor<double, 4>>("MO F12 Tensor", nobs, nobs, nri, nri);
    auto F2 = std::make_unique<Tensor<double, 4>>("MO F12_Squared Tensor", nobs, nobs, nobs, nri);
    auto FG = std::make_unique<Tensor<double, 4>>("MO F12G12 Tensor", nobs, nobs, nobs, nobs);
    auto Uf = std::make_unique<Tensor<double, 4>>("MO F12_DoubleCommutator Tensor", nobs, nobs, nobs, nobs);

    if (F12_TYPE == "DF") {
        // [J_AB]^{-1}(B|PQ)
        auto Metric = std::make_unique<Tensor<double, 3>>("Metric MO", naux, nri, nri);
        metric_ints(Metric.get(), bs, DFBS, nobs, naux, nri);

        for (int i = 0; i < teint.size(); i++){
            if ( teint[i] == "F" ){
                outfile->Printf("   F Integral\n");
                df_teints(teint[i], F.get(), Metric.get(), bs, DFBS, nobs, naux, nri, cgtg);
            } else if ( teint[i] == "FG" ){
                outfile->Printf("   FG Integral\n");
                df_teints(teint[i], FG.get(), Metric.get(), bs, DFBS, nobs, naux, nri, cgtg);
            } else if ( teint[i] == "F2" ){
                outfile->Printf("   F Squared Integral\n");
                df_teints(teint[i], F2.get(), Metric.get(), bs, DFBS, nobs, naux, nri, cgtg);
            } else if ( teint[i] == "Uf" ){
                outfile->Printf("   F Double Commutator Integral\n");
                df_teints(teint[i], Uf.get(), Metric.get(), bs, DFBS, nobs, naux, nri, cgtg);
            } else {
                outfile->Printf("   G Integral\n");
                df_teints(teint[i], G.get(), Metric.get(), bs, DFBS, nobs, naux, nri, cgtg);
            }
        }
    } else {
        for (int i = 0; i < teint.size(); i++){
            if ( teint[i] == "F" ){
                outfile->Printf("   F Integral\n");
                teints(teint[i], F.get(), bs, nobs, cgtg);
            } else if ( teint[i] == "FG" ){
                outfile->Printf("   FG Integral\n");
                teints(teint[i], FG.get(), bs, nobs, cgtg);
            } else if ( teint[i] == "F2" ){
                outfile->Printf("   F Squared Integral\n");
                teints(teint[i], F2.get(), bs, nobs, cgtg);
            } else if ( teint[i] == "Uf" ){
                outfile->Printf("   F Double Commutator Integral\n");
                teints(teint[i], Uf.get(), bs, nobs, cgtg);
            } else {
                outfile->Printf("   G Integral\n");
                teints(teint[i], G.get(), bs, nobs, cgtg);
            }
        }
    }

    /* Form the F12 Matrices */
    outfile->Printf("\n ===> Forming the F12 Intermediate Tensors <===\n");
    auto f = std::make_unique<Tensor<double, 2>>("Fock Matrix", nri, nri);
    auto k = std::make_unique<Tensor<double, 2>>("Exchange MO Integral", nri, nri);
    auto fk = std::make_unique<Tensor<double, 2>>("Fock-Exchange Matrix", nri, nri);
    auto V = std::make_unique<Tensor<double, 4>>("V Intermediate Tensor", nocc, nocc, nocc, nocc);
    auto X = std::make_unique<Tensor<double, 4>>("X Intermediate Tensor", nocc, nocc, nocc, nocc);
    auto C = std::make_unique<Tensor<double, 4>>("C Intermediate Tensor", nocc, nocc, nobs-nocc, nobs-nocc);
    auto B = std::make_unique<Tensor<double, 4>>("B Intermediate Tensor", nocc, nocc, nocc, nocc);
    auto D = std::make_unique<Tensor<double, 4>>("D Tensor", nocc, nocc, nobs-nocc, nobs-nocc);

    outfile->Printf("      V Intermediate\n");
    V_and_X_mat(V.get(), F.get(), G.get(), FG.get(), nocc, nobs, nri);
    FG.reset();
    outfile->Printf("   X Intermediate\n");
    V_and_X_mat(X.get(), F.get(), F.get(), F2.get(), nocc, nobs, nri);
    outfile->Printf("   Fock Matrix\n");
    f_mats(f.get(), k.get(), fk.get(), h.get(), G.get(), nocc, nri);
    h.reset();
    outfile->Printf("   C Intermediate\n");
    C_mat(C.get(), F.get(), f.get(), nocc, nobs, nri);
    outfile->Printf("   B Intermediate\n");
    B_mat(B.get(), Uf.get(), F2.get(), F.get(), f.get(), fk.get(), k.get(), nocc, nobs, ncabs, nri);
    Uf.reset();
    F2.reset();
    F.reset();
    fk.reset();
    D_mat(D.get(), f.get(), nocc, nobs);

    /* Compute the MP2F12/3C Energy */
    outfile->Printf("\n ===> Computing F12/3C(FIX) Energy Correction <===\n");

    auto E_f12 = compute_energy(G.get(), X.get(), B.get(), V.get(), f.get(), C.get(), D.get(),
                                nocc, nobs, nfrzn);

    double singles = 0.0;
    if (SINGLES == true) {
        singles = cabs_singles(f.get(), nocc, nri);
    }

    if (F12_TYPE == "DF") {
        outfile->Printf("\n ===> DF-MP2-F12/3C(FIX) Energies <===\n\n");
    } else {
        outfile->Printf("\n ===> MP2-F12/3C(FIX) Energies <===\n\n");
    }

    auto E_rhf = Process::environment.globals["CURRENT REFERENCE ENERGY"];
    auto E_mp2 = Process::environment.globals["CURRENT CORRELATION ENERGY"];

    if (F12_TYPE == "DF") {
        outfile->Printf("  Total DF-MP2-F12/3C(FIX) Energy:      %16.12f \n", E_rhf + E_mp2 + E_f12 + singles);
    } else {
        outfile->Printf("  Total MP2-F12/3C(FIX) Energy:         %16.12f \n", E_rhf + E_mp2 + E_f12 + singles);
    }
    outfile->Printf("     RHF Reference Energy:              %16.12f \n", E_rhf);
    outfile->Printf("     MP2 Correlation Energy:            %16.12f \n", E_mp2);
    outfile->Printf("     F12/3C(FIX) Correlation Energy:    %16.12f \n", E_f12);

    if (SINGLES == true) {
        outfile->Printf("     CABS Singles Correction:           %16.12f \n", singles);
    }

    // Typically you would build a new wavefunction and populate it with data
    return ref_wfn;
}

}} // End namespaces
