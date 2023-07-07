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

#include <psi4/libpsi4util/PsiOutStream.h>
#include <psi4/libqt/qt.h>

#include <psi4/libmints/basisset.h>
#include <psi4/libmints/dimension.h>
#include <psi4/libmints/integralparameters.h>
#include <psi4/libmints/mintshelper.h>
#include <psi4/libmints/orbitalspace.h>
#include <psi4/libmints/wavefunction.h>

#include "einsums/LinearAlgebra.hpp"
#include "einsums/Sort.hpp"
#include "einsums/Tensor.hpp"
#include "einsums/TensorAlgebra.hpp"

namespace psi { namespace mp2f12 {

MP2F12::MP2F12(SharedWavefunction reference_wavefunction, Options& options):
    Wavefunction(options) {
    reference_wavefunction_ = reference_wavefunction;
    common_init();
}

MP2F12::~MP2F12() {}

void MP2F12::common_init() 
{
    if (options_.get_str("REFERENCE") != "RHF") {
        throw PsiException("Only a restricted reference may be used",__FILE__,__LINE__);
    }
    
    print_ = options_.get_int("PRINT");
    singles_ = options_.get_bool("CABS_SINGLES");

    f12_type_ = options_.get_str("F12_TYPE");

    std::vector<OrbitalSpace> bs_ = {};
    nobs_ = reference_wavefunction_->basisset()->nbf();
    nocc_ = reference_wavefunction_->doccpi()[0];
    nvir_ = nobs_ - nocc_;
    nfrzn_ = reference_wavefunction_->frzcpi()[0];
    naux_ = 0;

    if ( f12_type_ == "DF" ) {
        DFBS_ = reference_wavefunction_->get_basisset("DF_BASIS_MP2");
        naux_ = DFBS_->nbf();
    }

    beta_ = options_.get_double("F12_BETA");
    cgtg_ = reference_wavefunction_->mintshelper()->f12_cgtg(beta_);

    nthreads_ = 1;
#ifdef _OPENMP
    nthreads_ = Process::environment.get_n_threads();
#endif
}

void MP2F12::print_header()
{
    outfile->Printf("\n -----------------------------------------------------------\n");
    if (f12_type_ == "DF") {
        outfile->Printf("                      DF-MP2-F12/3C(FIX)                    \n");
        outfile->Printf("       2nd Order Density-Fitted Explicitly Correlated       \n");
        outfile->Printf("                    Moeller-Plesset Theory                  \n");
        outfile->Printf("                RMP2 Wavefunction, %2d Threads              \n\n", nthreads_);
        outfile->Printf("               Erica Mitchell and Justin Turney             \n");
    } else {
        outfile->Printf("                       MP2-F12/3C(FIX)                      \n");
        outfile->Printf("   2nd Order Explicitly Correlated Moeller-Plesset Theory   \n");
        outfile->Printf("               RMP2 Wavefunction, %2d Threads               \n\n", nthreads_);
        outfile->Printf("              Erica Mitchell and Justin Turney              \n");
    }
    outfile->Printf(" -----------------------------------------------------------\n\n");
}

void MP2F12::form_basissets()
{
    outfile->Printf(" ===> Forming the OBS and CABS <===\n\n");

    outfile->Printf("  Orbital Basis Set (OBS)\n");
    OrbitalSpace OBS = reference_wavefunction_->alpha_orbital_space("p", "SO", "ALL");
    OBS.basisset()->print();

    outfile->Printf("  Complimentary Auxiliary Basis Set (CABS)\n");
    OrbitalSpace RI = OrbitalSpace::build_ri_space(reference_wavefunction_->get_basisset("CABS"), 1.0e-8);
    OrbitalSpace CABS = OrbitalSpace::build_cabs_space(OBS, RI, 1.0e-6);
    CABS.basisset()->print();
    
    if (f12_type_ == "DF") {
        outfile->Printf("  Auxiliary Basis Set\n");
        DFBS_->print();
    }

    nri_ = CABS.dim().max() + nobs_;
    ncabs_ = nri_ - nobs_;

    if (nfrzn_ != 0) {
        outfile->Printf("  Frozen Core Orbitals: %3d \n\n", nfrzn_);
    }

    outfile->Printf("  ----------------------------------------\n");
    outfile->Printf("     %5s  %5s   %5s  %5s  %5s   \n", "NOCC", "NOBS", "NCABS", "NRI", "NAUX");
    outfile->Printf("  ----------------------------------------\n");
    outfile->Printf("     %5d  %5d   %5d  %5d  %5d   \n", nocc_, nobs_, ncabs_, nri_, naux_);
    outfile->Printf("  ----------------------------------------\n");

    bs_ = {OBS, CABS};
}

void MP2F12::form_D(einsums::Tensor<double, 4> *D, einsums::Tensor<double, 2> *f)
{
    using namespace einsums;
    using namespace tensor_algebra;

#pragma omp parallel for collapse(4) num_threads(nthreads_)
    for (size_t i = 0; i < nocc_; i++) {
        for (size_t j = 0; j < nocc_; j++) {
            for (size_t a = nocc_; a < nobs_; a++) {
                for (size_t b = nocc_; b < nobs_; b++) {
                    auto denom = (*f)(a, a) + (*f)(b, b) - (*f)(i, i) - (*f)(j, j);
                    (*D)(i, j, a - nocc_, b - nocc_) = (1 / denom);
                }
            }
        }
    }
}

void MP2F12::form_cabs_singles(einsums::Tensor<double,2> *f)
{
    using namespace einsums;
    using namespace linear_algebra;

    int all_vir = nri_ - nocc_;

    // Diagonalize f_ij and f_AB
    Tensor<double, 2> C_ij{"occupied e-vecs", nocc_, nocc_};
    Tensor<double, 2> C_AB{"vir and CABS e-vecs", all_vir, all_vir};

    Tensor<double, 1> e_ij{"occupied e-vals", nocc_};
    Tensor<double, 1> e_AB{"vir and CABS e-vals", all_vir};
    {
        C_ij = (*f)(Range{0, nocc_}, Range{0, nocc_});
        C_AB = (*f)(Range{nocc_, nri_}, Range{nocc_, nri_});

        syev(&C_ij, &e_ij);
        syev(&C_AB, &e_AB);
    }

    // Form f_iA
    Tensor<double, 2> f_iA{"Fock Occ-All_vir", nocc_, all_vir};
    {
        Tensor f_view = (*f)(Range{0, nocc_}, Range{nocc_, nri_});

        gemm<false, false>(1.0, C_ij, gemm<false, true>(1.0, f_view, C_AB), 0.0, &f_iA);
    }

    double E_s = 0.0;
#pragma omp parallel for collapse(2) num_threads(nthreads_) reduction(+:E_s)
    for (size_t A = 0; A < all_vir; A++) {
        for (size_t i = 0; i < nocc_; i++) {
            E_s += 2 * pow(f_iA(i, A), 2) / (e_ij(i) - e_AB(A));
        }
    }

    E_singles_ = E_s;
}

void MP2F12::form_f12_energy(einsums::Tensor<double,4> *G, einsums::Tensor<double,4> *X, 
                               einsums::Tensor<double,4> *B, einsums::Tensor<double,4> *V,
                               einsums::Tensor<double,2> *f, einsums::Tensor<double,4> *C, 
                               einsums::Tensor<double,4> *D)
{
    using namespace einsums;
    using namespace tensor_algebra;
    using namespace tensor_algebra::index;

    auto E_f12_s = 0.0;
    auto E_f12_t = 0.0;
    int kd;

    auto G_ = (*G)(Range{0, nocc_}, Range{0, nocc_}, Range{nocc_, nobs_}, Range{nocc_, nobs_});

    outfile->Printf("  \n");
    outfile->Printf("  %1s   %1s  |     %14s     %14s     %12s \n",
                    "i", "j", "E_F12(Singlet)", "E_F12(Triplet)", "E_F12");
    outfile->Printf(" ----------------------------------------------------------------------\n");
    for (size_t i = nfrzn_; i < nocc_; i++) {
        for (size_t j = i; j < nocc_; j++) {
            // Allocations
            Tensor<double, 4> X_{"Scaled X", nocc_, nocc_, nocc_, nocc_};
            Tensor<double, 4> B_{"B ij", nocc_, nocc_, nocc_, nocc_};
            X_ = (*X)(Range{0, nocc_}, Range{0, nocc_}, Range{0, nocc_}, Range{0, nocc_});
            B_ = (*B)(Range{0, nocc_}, Range{0, nocc_}, Range{0, nocc_}, Range{0, nocc_});
            // Building B_
            auto f_scale = (*f)(i, i) + (*f)(j, j);
            linear_algebra::scale(f_scale, &X_);
            tensor_algebra::element([](double const &Bval, double const &Xval)
                                    -> double { return Bval - Xval; }, &B_, X_);
            // Getting V_Tilde and B_Tilde
            auto V_ = TensorView<double, 2>{(*V), Dim<2>{nocc_, nocc_}, Offset<4>{i, j, 0, 0},
                                            Stride<2>{(*V).stride(2), (*V).stride(3)}};
            auto K_ = TensorView<double, 2>{G_, Dim<2>{nvir_, nvir_}, Offset<4>{i, j, 0, 0},
                                            Stride<2>{G_.stride(2), G_.stride(3)}};
            auto D_ = TensorView<double, 2>{(*D), Dim<2>{nvir_, nvir_}, Offset<4>{i, j, 0, 0},
                                            Stride<2>{(*D).stride(2), (*D).stride(3)}};
            auto VT = V_Tilde(V_, C, K_, D_, i, j);
            auto BT = B_Tilde(B_, C, D_, i, j);
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

    E_f12_ = E_f12_s + E_f12_t;
}

double MP2F12::compute_energy()
{
    using namespace einsums;
    timer::initialize();

    print_header();

    /* Form the orbital spaces */
    timer_on("OBS and CABS");
    form_basissets();
    timer_off("OBS and CABS");

    outfile->Printf("\n ===> Forming the Integrals <===\n");

    /* Form the one-electron integrals */
    auto h = std::make_unique<Tensor<double, 2>>("MO One-Electron Integrals", nri_, nri_);
    timer_on("OEINTS");
    form_oeints(h.get());
    timer_off("OEINTS");

    /* Form the two-electron integrals */
    std::vector<std::string> teint = {"FG","Uf","G","F","F2"};

    // Two-Electron Integrals
    auto G = std::make_unique<Tensor<double, 4>>("MO G Tensor", nocc_, nocc_, nobs_, nri_);
    auto F = std::make_unique<Tensor<double, 4>>("MO F12 Tensor", nocc_, nocc_, nri_, nri_);
    auto F2 = std::make_unique<Tensor<double, 4>>("MO F12_Squared Tensor", nocc_, nocc_, nocc_, nri_);
    auto FG = std::make_unique<Tensor<double, 4>>("MO F12G12 Tensor", nocc_, nocc_, nocc_, nocc_);
    auto Uf = std::make_unique<Tensor<double, 4>>("MO F12_DoubleCommutator Tensor", nocc_, nocc_, nocc_, nocc_);

    // Fock Matrices
    auto f = std::make_unique<Tensor<double, 2>>("Fock Matrix", nri_, nri_);
    auto k = std::make_unique<Tensor<double, 2>>("Exchange MO Integral", nri_, nri_);
    auto fk = std::make_unique<Tensor<double, 2>>("Fock-Exchange Matrix", nri_, nri_);

    if (f12_type_ == "DF") {
        // [J_AB]^{-1}(B|PQ)
        auto Metric = std::make_unique<Tensor<double, 3>>("Metric MO", naux_, nri_, nri_);
        form_metric_ints(Metric.get());

        outfile->Printf("   Fock Matrix\n");
        timer_on("Fock Matrix");
        form_df_fock(f.get(), k.get(), fk.get(), h.get(), Metric.get());
        timer_off("Fock Matrix");
        h.reset();

        for (int i = 0; i < teint.size(); i++){
            if ( teint[i] == "F" ){
                outfile->Printf("   F Integral\n");
                timer_on("F_12 Integral");
                form_df_teints(teint[i], F.get(), Metric.get());
                timer_off("F_12 Integral");
            } else if ( teint[i] == "FG" ){
                outfile->Printf("   FG Integral\n");
                timer_on("FG_12 Integral");
                form_df_teints(teint[i], FG.get(), Metric.get());
                timer_off("FG_12 Integral");
            } else if ( teint[i] == "F2" ){
                outfile->Printf("   F Squared Integral\n");
                timer_on("F^2_12 Integral");
                form_df_teints(teint[i], F2.get(), Metric.get());
                timer_off("F^2_12 Integral");
            } else if ( teint[i] == "Uf" ){
                outfile->Printf("   F Double Commutator Integral\n");
                timer_on("U^F_12 Integral");
                form_df_teints(teint[i], Uf.get(), Metric.get());
                timer_off("U^F_12 Integral");
            } else {
                outfile->Printf("   G Integral\n");
                timer_on("G Integral");
                form_df_teints(teint[i], G.get(), Metric.get());
                timer_off("G Integral");
            }
        }
    } else {
        outfile->Printf("   Fock Matrix\n");
        timer_on("Fock Matrix");
        form_fock(f.get(), k.get(), fk.get(), h.get());
        timer_off("Fock Matrix");
        h.reset();

        for (int i = 0; i < teint.size(); i++){
            if ( teint[i] == "F" ){
                outfile->Printf("   F Integral\n");
                timer_on("F_12 Integral");
                form_teints(teint[i], F.get());
                timer_off("F_12 Integral");
            } else if ( teint[i] == "FG" ){
                outfile->Printf("   FG Integral\n");
                timer_on("FG_12 Integral");
                form_teints(teint[i], FG.get());
                timer_off("FG_12 Integral");
            } else if ( teint[i] == "F2" ){
                outfile->Printf("   F Squared Integral\n");
                timer_on("F^2_12 Integral");
                form_teints(teint[i], F2.get());
                timer_off("F^2_12 Integral");
            } else if ( teint[i] == "Uf" ){
                outfile->Printf("   F Double Commutator Integral\n");
                timer_on("U^F_12 Integral");
                form_teints(teint[i], Uf.get());
                timer_off("U^F_12 Integral");
            } else {
                outfile->Printf("   G Integral\n");
                timer_on("G Integral");
                form_teints(teint[i], G.get());
                timer_off("G Integral");
            }
        }
    }

    /* Form the F12 Matrices */
    outfile->Printf("\n ===> Forming the F12 Intermediate Tensors <===\n");
    auto V = std::make_unique<Tensor<double, 4>>("V Intermediate Tensor", nocc_, nocc_, nocc_, nocc_);
    auto X = std::make_unique<Tensor<double, 4>>("X Intermediate Tensor", nocc_, nocc_, nocc_, nocc_);
    auto C = std::make_unique<Tensor<double, 4>>("C Intermediate Tensor", nocc_, nocc_, nobs_ - nocc_, nobs_ - nocc_);
    auto B = std::make_unique<Tensor<double, 4>>("B Intermediate Tensor", nocc_, nocc_, nocc_, nocc_);
    auto D = std::make_unique<Tensor<double, 4>>("D Tensor", nocc_, nocc_, nobs_ - nocc_, nobs_ - nocc_);

    outfile->Printf("   V Intermediate\n");
    timer_on("V Intermediate");
    form_V_or_X(V.get(), F.get(), G.get(), FG.get());
    timer_off("V Intermediate");
    FG.reset();

    outfile->Printf("   X Intermediate\n");
    timer_on("X Intermediate");
    form_V_or_X(X.get(), F.get(), F.get(), F2.get());
    timer_off("X Intermediate");

    outfile->Printf("   C Intermediate\n");
    timer_on("C Intermediate");
    form_C(C.get(), F.get(), f.get());
    timer_off("C Intermediate");

    outfile->Printf("   B Intermediate\n");
    timer_on("B Intermediate");
    form_B(B.get(), Uf.get(), F2.get(), F.get(), f.get(), fk.get(), k.get());
    timer_off("B Intermediate");
    Uf.reset();
    F2.reset();
    F.reset();
    fk.reset();

    timer_on("Energy Denom");
    form_D(D.get(), f.get());
    timer_off("Energy Denom");

    /* Compute the MP2F12/3C Energy */
    outfile->Printf("\n ===> Computing F12/3C(FIX) Energy Correction <===\n");
    timer_on("F12 Energy Correction");
    form_f12_energy(G.get(), X.get(), B.get(), V.get(), f.get(), C.get(), D.get());
    timer_off("F12 Energy Correction");

    if (singles_ == true) {
        timer_on("CABS Singles Correction");
        form_cabs_singles(f.get());
        timer_off("CABS Singles Correction");
    }

    print_results();
    //timer::report();
    timer::finalize();

    // Typically you would build a new wavefunction and populate it with data
    return E_f12_total_;
}

void MP2F12::print_results()
{
    if (f12_type_ == "DF") {
        outfile->Printf("\n ===> DF-MP2-F12/3C(FIX) Energies <===\n\n");
    } else {
        outfile->Printf("\n ===> MP2-F12/3C(FIX) Energies <===\n\n");
    }

    auto E_rhf = Process::environment.globals["CURRENT REFERENCE ENERGY"];
    auto E_mp2 = Process::environment.globals["CURRENT CORRELATION ENERGY"];

    E_f12_total_ = E_rhf + E_mp2 + E_f12_ + E_singles_;

    if (f12_type_ == "DF") {
        outfile->Printf("  Total DF-MP2-F12/3C(FIX) Energy:      %16.12f \n", E_f12_total_);
    } else {
        outfile->Printf("  Total MP2-F12/3C(FIX) Energy:         %16.12f \n", E_f12_total_);
    }
    outfile->Printf("     RHF Reference Energy:              %16.12f \n", E_rhf);
    outfile->Printf("     MP2 Correlation Energy:            %16.12f \n", E_mp2);
    outfile->Printf("     F12/3C(FIX) Correlation Energy:    %16.12f \n", E_f12_);

    if (singles_ == true) {
        outfile->Printf("     CABS Singles Correction:           %16.12f \n", E_singles_);
    }

    set_scalar_variable("CURRENT CORRELATION ENERGY", E_f12_);
}

double MP2F12::t_(const int& p, const int& q, const int& r, const int& s)
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

std::pair<double, double> MP2F12::V_Tilde(einsums::TensorView<double, 2>& V_, einsums::Tensor<double, 4> *C,
                                          einsums::TensorView<double, 2>& K_ij, einsums::TensorView<double, 2>& D_ij,
                                          const int& i, const int& j)
{
    using namespace einsums;
    using namespace tensor_algebra;
    using namespace tensor_algebra::index;

    auto V_s = 0.0;
    auto V_t = 0.0;
    int kd;

    auto V_ij = std::make_unique<Tensor<double, 2>>("V(i, j, :, :)", nocc_, nocc_);
    auto KD = std::make_unique<Tensor<double, 2>>("Temp 1", nvir_, nvir_);
    (*V_ij) = V_;

    einsum(Indices{a, b}, &KD, Indices{a, b}, K_ij, Indices{a, b}, D_ij);
    einsum(1.0, Indices{k, l}, &V_ij, -1.0, Indices{k, l, a, b}, *C, Indices{a, b}, KD);
    ( i == j ) ? ( kd = 1 ) : ( kd = 2 );

    V_s += 0.5 * (t_(i, j, i, j) + t_(i, j, j, i)) * kd * ((*V_ij)(i, j) + (*V_ij)(j, i));

    if ( i != j ) {
        V_t += 0.5 * (t_(i, j, i, j) - t_(i, j, j, i)) * kd * ((*V_ij)(i, j) - (*V_ij)(j, i));
    }
    return {V_s, V_t};
}

std::pair<double, double> MP2F12::B_Tilde(einsums::Tensor<double, 4>& B, einsums::Tensor<double, 4> *C, 
                                          einsums::TensorView<double, 2>& D_ij, 
                                          const int& i, const int& j)
{
    using namespace einsums;
    using namespace tensor_algebra;
    using namespace tensor_algebra::index;

    auto B_s = 0.0;
    auto B_t = 0.0;
    int kd;

    Tensor<double, 4> B_ij{"B = B - X * (fii +fjj)", nocc_, nocc_, nocc_, nocc_};
    Tensor<double, 4> CD{"Temp 1", nocc_, nocc_, nvir_, nvir_};
    B_ij = B;
    ( i == j ) ? ( kd = 1 ) : ( kd = 2 );

    einsum(Indices{k, l, a, b}, &CD, Indices{k, l, a, b}, *C, Indices{a, b}, D_ij);
    einsum(1.0, Indices{k, l, m, n}, &B_ij, -1.0, Indices{m, n, a, b}, *C,
                                                  Indices{k, l, a, b}, CD);

    B_s += 0.125 * (t_(i, j, i, j) + t_(i, j, j, i)) * kd 
                 * (B_ij(i, j, i, j) + B_ij(j, i, i, j))
                 * (t_(i, j, i, j) + t_(i, j, j, i)) * kd;

    if ( i != j ) {
        B_t += 0.125 * (t_(i, j, i, j) - t_(i, j, j, i)) * kd
                     * (B_ij(i, j, i, j) - B_ij(j, i, i, j))
                     * (t_(i, j, i, j) - t_(i, j, j, i)) * kd;
    }
    return {B_s, B_t};
}

}} // End namespaces
