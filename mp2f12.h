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

#ifndef MP2F12_H
#define MP2F12_H

#include <psi4/libmints/basisset.h>
#include <psi4/libmints/integralparameters.h>
#include <psi4/libmints/onebody.h>
#include <psi4/libmints/orbitalspace.h>
#include <psi4/libmints/wavefunction.h>
#include <psi4/liboptions/liboptions.h>

#include "einsums/Tensor.hpp"

namespace psi { namespace mp2f12 {

class MP2F12 : public Wavefunction {
   public: 
    MP2F12(SharedWavefunction reference_wavefunction, Options& options);
    ~MP2F12();

    /* Compute the total MP2-F12/3C(FIX) Energy */
    double compute_energy();

   protected:
    /* Print level */
    int print_;

    /* Number of OMP_THREADS */
    int nthreads_;

    /* Choose to compute CABS Singles correction */
    bool singles_;

    /* Choose CONV or DF F12 computation */
    std::string f12_type_;

    /* Density-fitting Basis Set (DFBS) */
    std::shared_ptr<BasisSet> DFBS_;

    /* List of orbital spaces: Orbital Basis Set (OBS) 
       and Complimentary Auxiliary Basis Set (CABS) */
    std::vector<OrbitalSpace> bs_;

    /* Number of basis functions in OBS */
    int nobs_;

    /* Number of basis functions in CABS */
    int ncabs_;

    /* Number of basis functions in total */
    int nri_;

    /* Number of occupied orbitals */
    int nocc_;

    /* Number of virtual orbitals */
    int nvir_;

    /* Number of basis functions in DFBS */
    int naux_;

    /* Number of frozen core orbitals */
    int nfrzn_;

    /* F12 Correlation Factor, Contracted Gaussian-Type Geminal */
    std::shared_ptr<CorrelationFactor> cgtg_;

    /* $\beta$ in F12 CGTG */
    double beta_;

    /* F12 energy */
    double E_f12_;

    /* CABS Singles Correction */
    double E_singles_;

    /* Total MP2-F12/3C(FIX) Energy */
    double E_f12_total_;

    /* Form the basis sets OBS and CABS */
    void form_basissets();

    /* Form the energy denominator */
    void form_D(einsums::Tensor<double, 4> *D, einsums::Tensor<double, 2> *f);

    /* Form the CABS Singles correction $\frac{|f^{a'}_{i}}|^2}{e_{a'} - e_{i}}$ */
    void form_cabs_singles(einsums::Tensor<double,2> *f);

    /* Form the F12/3C(FIX) correlation energy */
    void form_f12_energy(einsums::Tensor<double,4> *G, einsums::Tensor<double,4> *X, 
                         einsums::Tensor<double,4> *B, einsums::Tensor<double,4> *V,
                         einsums::Tensor<double,2> *f, einsums::Tensor<double,4> *C, 
                         einsums::Tensor<double,4> *D);
   
    /* Form the one-electron integrals H = T + V */
    void form_oeints(einsums::Tensor<double, 2> *h);

    /* Form the convetional two-electron integrals */
    void form_teints(const std::string& int_type, einsums::Tensor<double, 4> *ERI);
    
    /* Form the density-fitted two-electron integrals */
    void form_df_teints(const std::string& int_type, einsums::Tensor<double, 4> *ERI, einsums::Tensor<double, 3> *Metric);
    
    /* Form the Fock matrix */
    void form_fock(einsums::Tensor<double, 2> *f, einsums::Tensor<double, 2> *k, 
                   einsums::Tensor<double, 2> *fk, einsums::Tensor<double, 2> *h,
                   einsums::Tensor<double, 4> *G);

    /* Form the $V^{ij}_{kl}$ or $X^{ij}_{kl}$ tensor */
    void form_V_or_X(einsums::Tensor<double, 4> *VX, einsums::Tensor<double, 4> *F,
                     einsums::Tensor<double, 4> *G_F, einsums::Tensor<double, 4> *FG_F2);
    
    /* Form the $C^{kl}_{ab}$ tensor */
    void form_C(einsums::Tensor<double, 4> *C, einsums::Tensor<double, 4> *F,
                einsums::Tensor<double, 2> *f);
    
    /* Form the $B^{kl}_{mn}$ tensor */
    void form_B(einsums::Tensor<double, 4> *B, einsums::Tensor<double, 4> *Uf,
                einsums::Tensor<double, 4> *F2, einsums::Tensor<double, 4> *F,
                einsums::Tensor<double, 2> *f, einsums::Tensor<double, 2> *fk,
                einsums::Tensor<double, 2> *kk);

   private:
    void common_init();

    void print_header();
    void print_results();

    /* Returns the fixed amplitudes value */
    double t_(const int& p, const int& q, const int& r, const int& s);

    /* Form the $T^{ij}_{ij}\Tilde{V}^{ij}_{ij}$ contirbution to the energy */
    std::pair<double, double> V_Tilde(einsums::TensorView<double, 2>& V_, einsums::Tensor<double, 4> *C,
                                      einsums::TensorView<double, 2>& K_ij, einsums::TensorView<double, 2>& D_ij,
                                      const int& i, const int& j);

    /* Form the $T^{ij}_{ij}\Tilde{B}^{ij}_{ij}T^{ij}_{ij}$ contirbution to the energy */
    std::pair<double, double> B_Tilde(einsums::Tensor<double, 4>& B, einsums::Tensor<double, 4> *C, 
                                      einsums::TensorView<double, 2>& D_ij, 
                                      const int& i, const int& j);

    void one_body_ao_computer(std::vector<std::shared_ptr<OneBodyAOInt>> ints, SharedMatrix M, bool symm);

    SharedMatrix ao_kinetic(std::shared_ptr<BasisSet> bs1, std::shared_ptr<BasisSet> bs2);

    SharedMatrix ao_potential(std::shared_ptr<BasisSet> bs1, std::shared_ptr<BasisSet> bs2);

    void convert_C(einsums::Tensor<double,2> *C, OrbitalSpace bs);

    void set_ERI(einsums::TensorView<double, 4>& ERI_Slice, einsums::Tensor<double, 4> *Slice);
    void set_ERI(einsums::TensorView<double, 3>& ERI_Slice, einsums::Tensor<double, 3> *Slice);

    /* Form the integrals containing the DF metric [J_AB]^{-1}(B|PQ) */
    void form_metric_ints(einsums::Tensor<double, 3> *DF_ERI);
    
    /* Form the integrals containing the explicit correlation (B|\hat{A}_{12}|PQ) */
    void form_oper_ints(const std::string& int_type, einsums::Tensor<double, 3> *DF_ERI, einsums::Tensor<double, 2> *AB);
};

}} // end namespaces
#endif