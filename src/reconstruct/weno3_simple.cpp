//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file weno3_simple.cpp
//! \brief WENO reconstruction (3rd order accuracy)
//! Operates on the entire nx4 range of a single AthenaArray<Real> input (no MHD).
//! No assumptions of hydrodynamic fluid variable input; no characteristic projection.
//!
//! REFERENCES:
//! - (Mignone) A. Mignone, "High-order conservative reconstruction schemes for finite
//!   volume methods in cylindrical and spherical coordinates", JCP, 270, 784 (2014)
//========================================================================================

// C headers

// C++ headers
#include <algorithm>    // max()

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../coordinates/coordinates.hpp"
#include "../eos/eos.hpp"
#include "../mesh/mesh.hpp"
#include "reconstruction.hpp"

//----------------------------------------------------------------------------------------
//! \fn Reconstruction::Weno3X1(const int k, const int j,
//!                             const int il, const int iu,
//!                             const AthenaArray<Real> &w, const AthenaArray<Real> &bcc,
//!                             AthenaArray<Real> &wl, AthenaArray<Real> &wr)
//! \brief Returns L/R interface values in X1-dir constructed using third-order WENO
//!        over [kl,ku][jl,ju][il,iu]

void Reconstruction::Weno3X1(const int k, const int j, const int il, const int iu,
          const AthenaArray<Real> &q, AthenaArray<Real> &ql, AthenaArray<Real> &qr) {
  Coordinates *pco = pmy_block_->pcoord;
  const int nu = q.GetDim4() - 1;
  const Real cref_over_n = 20.0/static_cast<Real>(pmy_block_->pmy_mesh->mesh_size.nx1);

  // set work arrays used for primitive/characterstic cell-averages to scratch
  AthenaArray<Real> &q_im1 = scr1_ni_, &q_i = scr2_ni_,
                    &q_ip1 = scr3_ni_, &qr_imh = scr4_ni_, &ql_iph = scr5_ni_;

  // set work WENO arrays to shallow copies of scratch arrays:
  AthenaArray<Real> &dqif = scr01_i_, &dqib = scr02_i_, &qiref = scr03_i_,
                    &qipf = scr04_i_, &qipb = scr05_i_, qimf = scr06_i_, &qimb = scr07_i_,
                    &omega_p0 = scr08_i_, &omega_m0 = scr09_i_,
                    &qr_plm = scr10_i_, &ql_plm = scr11_i_,
                    &dip0 = scr12_i_, &dim0 = scr13_i_;

  // cache the x1-sliced primitive states for eigensystem calculation
  for (int n=0; n<=nu; ++n) {
#pragma omp simd
    for (int i=il; i<=iu; ++i) {
      q_im1(n,i) = q(n,k,j,i-1);
      q_i  (n,i) = q(n,k,j,i  );
      q_ip1(n,i) = q(n,k,j,i+1);
    }
  }

  for (int n=0; n<=nu; ++n) {
    //--- Step 1. ------------------------------------------------------------------------
    // Calculate interface averages using PLM (Mignone eq 28, 29)
    // nonuniform or uniform Cartesian-like coord reconstruction from volume averages:
    if (uniform[X1DIR] && !curvilinear[X1DIR]) {
#pragma omp simd
      for (int i=il; i<=iu; ++i) {
        dqif(i) = q_ip1(n,i)-q_i(n,i);
        dqib(i) = q_i(n,i)-q_im1(n,i);
      }
#pragma omp simd simdlen(SIMD_WIDTH)
      for (int i=il; i<=iu; ++i) {
        qipf(i) = q_i(n,i) + 0.5*dqif(i);
        qipb(i) = q_i(n,i) + 0.5*dqib(i);
        qimf(i) = q_i(n,i) - 0.5*dqif(i);
        qimb(i) = q_i(n,i) - 0.5*dqib(i);
      }
    } else {
#pragma omp simd simdlen(SIMD_WIDTH)
      for (int i=il; i<=iu; ++i) {
        dqif(i) = (q_ip1(n,i)-q_i(n,i))*pco->dx1f(i)/pco->dx1v(i);
        dqib(i) = (q_i(n,i)-q_im1(n,i))*pco->dx1f(i)/pco->dx1v(i-1);
      }
#pragma omp simd simdlen(SIMD_WIDTH)
      for (int i=il; i<=iu; ++i) {
        Real dx = pco->dx1f(i);
        qipf(i) = q_i(n,i) + dqif(i)*(pco->x1f(i+1)-pco->x1v(i))/dx;
        qipb(i) = q_i(n,i) + dqib(i)*(pco->x1f(i+1)-pco->x1v(i))/dx;
        qimf(i) = q_i(n,i) - dqif(i)*(pco->x1v(i  )-pco->x1f(i))/dx;
        qimb(i) = q_i(n,i) - dqib(i)*(pco->x1v(i  )-pco->x1f(i))/dx;
      }
    }

    //--- Step 2. ------------------------------------------------------------------------
    // Calculate qi_ref (Mignone eq 41)
#pragma omp simd simdlen(SIMD_WIDTH)
    for (int i=il; i<=iu; ++i) {
      Real a1 = std::abs(q_i(n,i));
      Real a2 = std::abs(q_im1(n,i));
      Real a3 = std::abs(q_ip1(n,i));
      qiref(i) = cref_over_n*std::max(a1,std::max(a2, a3));
    }

    //--- Step 3. ------------------------------------------------------------------------
    // Calculate d, alpha, & omega values (Mignone eq 40, 42)
    if (uniform[X1DIR] && !curvilinear[X1DIR]) {
#pragma omp simd simdlen(SIMD_WIDTH)
      for (int i=il; i<=iu; ++i) {
        dip0(i) =  2.0*c1i(i); // c1i = wp1
        dim0(i) = -2.0*c2i(i); // c2i = wm1
      }
    } else {
#pragma omp simd simdlen(SIMD_WIDTH)
      for (int i=il; i<=iu; ++i) {
        dip0(i) =  c1i(i)*pco->dx1v(i)/(pco->x1f(i+1)-pco->x1v(i)); // c2i = wp1
        dim0(i) = -c2i(i)*pco->dx1v(i)/(pco->x1v(i  )-pco->x1f(i)); // c1i = wm1
      }
    }

#pragma omp simd simdlen(SIMD_WIDTH)
    for (int i=il; i<=iu; ++i) {
      Real b0 = (1.0+SQR(dqif(i)-dqib(i))/(SQR(dqif(i))+SQR(qiref(i))));
      Real b1 = (1.0+SQR(dqif(i)-dqib(i))/(SQR(dqib(i))+SQR(qiref(i))));
      Real alpha_p0 = dip0(i)*b0;
      Real alpha_p1 = (1.0-dip0(i))*b1;
      Real alpha_m0 = dim0(i)*b0;
      Real alpha_m1 = (1.0-dim0(i))*b1;
      omega_p0(i) = alpha_p0/(alpha_p0+alpha_p1);
      omega_m0(i) = alpha_m0/(alpha_m0+alpha_m1);
    }

    //--- Step 4. ------------------------------------------------------------------------
    // Convert limited cell-centered values to interface-centered L/R Riemann states
    // both L/R values defined over [il,iu]
#pragma omp simd simdlen(SIMD_WIDTH)
    for (int i=il; i<=iu; ++i) {
      ql_iph(n,i) = omega_p0(i)*qipf(i)+(1.0-omega_p0(i))*qipb(i);
      qr_imh(n,i) = omega_m0(i)*qimf(i)+(1.0-omega_m0(i))*qimb(i);
    }

    if (uniform[X1DIR] && !curvilinear[X1DIR]) {
#pragma omp simd simdlen(SIMD_WIDTH)
      for (int i=il; i<=iu; ++i) {
        Real dq2 = dqif(i)*dqib(i);
        Real dwm = 2.0*dq2/(dqif(i)+dqib(i));
        if (dq2 <= 0.0) dwm = 0.0;
        ql_plm(i) = q_i(n,i) + 0.5*dwm;
        qr_plm(i) = q_i(n,i) - 0.5*dwm;
      }
    } else {
#pragma omp simd simdlen(SIMD_WIDTH)
      for (int i=il; i<=iu; ++i) {
        Real dq2 = dqif(i)*dqib(i);
        Real cf = pco->dx1v(i  )/(pco->x1f(i+1) - pco->x1v(i)); // (Mignone eq 33)
        Real cb = pco->dx1v(i-1)/(pco->x1v(i  ) - pco->x1f(i));
        Real dwm = (dq2*(cf*dqib(i)+cb*dqif(i))/
                    (SQR(dqib(i))+SQR(dqif(i))+dq2*(cf+cb-2.0)));
        if (dq2 <= 0.0) dwm = 0.0;
        ql_plm(i) = q_i(n,i) + ((pco->x1f(i+1) - pco->x1v(i))/pco->dx1f(i))*dwm;
        qr_plm(i) = q_i(n,i) - ((pco->x1v(i  ) - pco->x1f(i))/pco->dx1f(i))*dwm;
      }
    }

#pragma omp simd simdlen(SIMD_WIDTH)
    for (int i=il; i<=iu; ++i) {
      Real plcheck = 0.5*(1.0+SIGN(ql_iph(n,i)));
      Real prcheck = 0.5*(1.0+SIGN(qr_imh(n,i)));
      ql_iph(n,i) = plcheck*ql_iph(n,i)+(1.0-plcheck)*ql_plm(i);
      qr_imh(n,i) = prcheck*qr_imh(n,i)+(1.0-prcheck)*qr_plm(i);
    }
  } // end char WENO3 loop over =nu

  // compute ql_(i+1/2) and qr_(i-1/2)
  for (int n=0; n<=nu; ++n) {
#pragma omp simd
    for (int i=il; i<=iu; ++i) {
      ql(n,i+1) = ql_iph(n,i);
      qr(n,i  ) = qr_imh(n,i);
    }
  }
  return;
}

//-------------------------------------------------------------------------------------
//! \fn Reconstruction::Weno3X2(const int k, const int j,
//!                             const int il, const int iu,
//!                             const AthenaArray<Real> &q,
//!                             AthenaArray<Real> &ql, AthenaArray<Real> &qr)
//! \brief Returns L/R interface values in X2-dir constructed using third-order WENO
//!        over [kl,ku][jl,ju][il,iu]

void Reconstruction::Weno3X2(const int k, const int j, const int il, const int iu,
          const AthenaArray<Real> &q, AthenaArray<Real> &ql, AthenaArray<Real> &qr) {
  Coordinates *pco = pmy_block_->pcoord;
  const int nu = q.GetDim4() - 1;
  const Real cref_over_n = 20.0/static_cast<Real>(pmy_block_->pmy_mesh->mesh_size.nx2);

  // set work arrays used for primitive/characterstic cell-averages to scratch
  AthenaArray<Real> &q_jm1 = scr1_ni_, &q_j = scr2_ni_,
                    &q_jp1 = scr3_ni_, &qr_jmh = scr4_ni_, &ql_jph = scr5_ni_;

  // set work WENO arrays to shallow copies of scratch arrays:
  AthenaArray<Real> &dqjf = scr01_i_, &dqjb = scr02_i_, &qjref = scr03_i_,
                    &qjpf = scr04_i_, &qjpb = scr05_i_, qjmf = scr06_i_, &qjmb = scr07_i_,
                    &omega_p0 = scr08_i_, &omega_m0 = scr09_i_,
                    &qr_plm = scr10_i_, &ql_plm = scr11_i_;

  // cache the x1-sliced primitive states for eigensystem calculation
  for (int n=0; n<=nu; ++n) {
#pragma omp simd
    for (int i=il; i<=iu; ++i) {
      q_jm1(n,i) = q(n,k,j-1,i);
      q_j  (n,i) = q(n,k,j  ,i);
      q_jp1(n,i) = q(n,k,j+1,i);
    }
  }

  for (int n=0; n<=nu; ++n) {
    //--- Step 1. ------------------------------------------------------------------------
    // Calculate interface averages using PLM (Mignone eq 28, 29)
    if (uniform[X2DIR] && !curvilinear[X2DIR]) {
#pragma omp simd
      for (int i=il; i<=iu; ++i) {
        dqjf(i) = q_jp1(n,i)-q_j(n,i);
        dqjb(i) = q_j(n,i)-q_jm1(n,i);
      }
#pragma omp simd simdlen(SIMD_WIDTH)
      for (int i=il; i<=iu; ++i) {
        qjpf(i) = q_j(n,i) + 0.5*dqjf(i);
        qjpb(i) = q_j(n,i) + 0.5*dqjb(i);
        qjmf(i) = q_j(n,i) - 0.5*dqjf(i);
        qjmb(i) = q_j(n,i) - 0.5*dqjb(i);
      }
    } else {
#pragma omp simd simdlen(SIMD_WIDTH)
      for (int i=il; i<=iu; ++i) {
        dqjf(i) = (q_jp1(n,i)-q_j(n,i))*pco->dx2f(j)/pco->dx2v(j);
        dqjb(i) = (q_j(n,i)-q_jm1(n,i))*pco->dx2f(j)/pco->dx2v(j-1);
      }
#pragma omp simd simdlen(SIMD_WIDTH)
      for (int i=il; i<=iu; ++i) {
        Real dx = pco->dx2f(j);
        qjpf(i) = q_j(n,i) + dqjf(i)*(pco->x2f(j+1)-pco->x2v(j))/dx;
        qjpb(i) = q_j(n,i) + dqjb(i)*(pco->x2f(j+1)-pco->x2v(j))/dx;
        qjmf(i) = q_j(n,i) - dqjf(i)*(pco->x2v(j  )-pco->x2f(j))/dx;
        qjmb(i) = q_j(n,i) - dqjb(i)*(pco->x2v(j  )-pco->x2f(j))/dx;
      }
    }

    //--- Step 2. ------------------------------------------------------------------------
    // Calculate qi_ref (Mignone eq 41)
#pragma omp simd simdlen(SIMD_WIDTH)
    for (int i=il; i<=iu; ++i) {
      Real a1 = std::abs(q_j(n,i));
      Real a2 = std::abs(q_jm1(n,i));
      Real a3 = std::abs(q_jp1(n,i));
      qjref(i) = cref_over_n*std::max(a1,std::max(a2, a3));
    }

    //--- Step 3. ------------------------------------------------------------------------
    // Calculate d, alpha, & omega values (Mignone eq 40, 42)
    Real djp0, djm0;
    if (uniform[X2DIR] && !curvilinear[X1DIR]) {
      djp0 =  2.0*c1j(j); // c1j = wp1
      djm0 = -2.0*c2j(j); // c2j = wm1
    } else {
      djp0 =  c1j(j)*pco->dx2v(j)/(pco->x2f(j+1)-pco->x2v(j)); // c2i = wp1
      djm0 = -c2j(j)*pco->dx2v(j)/(pco->x2v(j  )-pco->x2f(j)); // c1i = wm1
    }

#pragma omp simd simdlen(SIMD_WIDTH)
    for (int i=il; i<=iu; ++i) {
      Real b0 = (1.0+SQR(dqjf(i)-dqjb(i))/(SQR(dqjf(i))+SQR(qjref(i))));
      Real b1 = (1.0+SQR(dqjf(i)-dqjb(i))/(SQR(dqjb(i))+SQR(qjref(i))));
      Real alpha_p0 = djp0*b0;
      Real alpha_p1 = (1.0-djp0)*b1;
      Real alpha_m0 = djm0*b0;
      Real alpha_m1 = (1.0-djm0)*b1;
      omega_p0(i) = alpha_p0/(alpha_p0+alpha_p1);
      omega_m0(i) = alpha_m0/(alpha_m0+alpha_m1);
    }

    //--- Step 4. ------------------------------------------------------------------------
    // Convert limited cell-centered values to interface-centered L/R Riemann states
    // both L/R values defined over [il,iu]
#pragma omp simd simdlen(SIMD_WIDTH)
    for (int i=il; i<=iu; ++i) {
      ql_jph(n,i) = omega_p0(i)*qjpf(i)+(1.0-omega_p0(i))*qjpb(i);
      qr_jmh(n,i) = omega_m0(i)*qjmf(i)+(1.0-omega_m0(i))*qjmb(i);
    }

    if (uniform[X2DIR] && !curvilinear[X2DIR]) {
#pragma omp simd simdlen(SIMD_WIDTH)
      for (int i=il; i<=iu; ++i) {
        Real dq2 = dqjf(i)*dqjb(i);
        Real dwm = 2.0*dq2/(dqjf(i)+dqjb(i));
        if (dq2 <= 0.0) dwm = 0.0;
        ql_plm(i) = q_j(n,i) + 0.5*dwm;
        qr_plm(i) = q_j(n,i) - 0.5*dwm;
      }
    } else {
#pragma omp simd simdlen(SIMD_WIDTH)
      for (int i=il; i<=iu; ++i) {
        Real dq2 = dqjf(i)*dqjb(i);
        Real cf = pco->dx2v(j  )/(pco->x2f(j+1) - pco->x2v(j)); // (Mignone eq 33)
        Real cb = pco->dx2v(j-1)/(pco->x2v(j  ) - pco->x2f(j));
        Real dwm = (dq2*(cf*dqjb(i)+cb*dqjf(i))/
                    (SQR(dqjb(i))+SQR(dqjf(i))+dq2*(cf+cb-2.0)));
        if (dq2 <= 0.0) dwm = 0.0;
        ql_plm(i) = q_j(n,i) + ((pco->x2f(j+1) - pco->x2v(j))/pco->dx2f(j))*dwm;
        qr_plm(i) = q_j(n,i) - ((pco->x2v(j  ) - pco->x2f(j))/pco->dx2f(j))*dwm;
      }
    }

#pragma omp simd simdlen(SIMD_WIDTH)
    for (int i=il; i<=iu; ++i) {
      Real plcheck = 0.5*(1.0+SIGN(ql_jph(n,i)));
      Real prcheck = 0.5*(1.0+SIGN(qr_jmh(n,i)));
      ql_jph(n,i) = plcheck*ql_jph(n,i)+(1.0-plcheck)*ql_plm(i);
      qr_jmh(n,i) = prcheck*qr_jmh(n,i)+(1.0-prcheck)*qr_plm(i);
    }
  } // end char WENO3 loop over =nu


  // compute ql_(j+1/2) and qr_(j-1/2)
  for (int n=0; n<=nu; ++n) {
#pragma omp simd
    for (int i=il; i<=iu; ++i) {
      ql(n,i) = ql_jph(n,i);
      qr(n,i) = qr_jmh(n,i);
    }
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn Reconstruction::Weno3X3(const int k, const int j,
//!                             const int il, const int iu,
//!                             const AthenaArray<Real> &w, const AthenaArray<Real> &bcc,
//!                             AthenaArray<Real> &wl, AthenaArray<Real> &wr)
//! \brief Returns L/R interface values in X3-dir constructed using third-order WENO
//!        over [kl,ku][jl,ju][il,iu]

void Reconstruction::Weno3X3(const int k, const int j, const int il, const int iu,
         const AthenaArray<Real> &q, AthenaArray<Real> &ql, AthenaArray<Real> &qr) {
  Coordinates *pco = pmy_block_->pcoord;
  const int nu = q.GetDim4() - 1;
  const Real cref_over_n = 20.0/static_cast<Real>(pmy_block_->pmy_mesh->mesh_size.nx3);

  // set work arrays used for primitive/characterstic cell-averages to scratch
  AthenaArray<Real> &q_km1 = scr1_ni_, &q_k = scr2_ni_,
                    &q_kp1 = scr3_ni_, &qr_kmh = scr4_ni_, &ql_kph = scr5_ni_;

  // set work WENO arrays to shallow copies of scratch arrays:
  AthenaArray<Real> &dqkf = scr01_i_, &dqkb = scr02_i_, &qkref = scr03_i_,
                    &qkpf = scr04_i_, &qkpb = scr05_i_, qkmf = scr06_i_, &qkmb = scr07_i_,
                    &omega_p0 = scr08_i_, &omega_m0 = scr09_i_,
                    &qr_plm = scr10_i_, &ql_plm = scr11_i_;

  // cache the x1-sliced primitive states for eigensystem calculation
  for (int n=0; n<=nu; ++n) {
#pragma omp simd
    for (int i=il; i<=iu; ++i) {
      q_km1(n,i) = q(n,k-1,j,i);
      q_k  (n,i) = q(n,k  ,j,i);
      q_kp1(n,i) = q(n,k+1,j,i);
    }
  }

  for (int n=0; n<=nu; ++n) {
    //--- Step 1. ------------------------------------------------------------------------
    // Calculate interface averages using PLM (Mignone eq 28, 29)
#pragma omp simd
    for (int i=il; i<=iu; ++i) {
      dqkf(i) = q_kp1(n,i)-q_k(n,i);
      dqkb(i) = q_k(n,i)-q_km1(n,i);
    }
#pragma omp simd simdlen(SIMD_WIDTH)
    for (int i=il; i<=iu; ++i) {
      qkpf(i) = q_k(n,i) + 0.5*dqkf(i);
      qkpb(i) = q_k(n,i) + 0.5*dqkb(i);
      qkmf(i) = q_k(n,i) - 0.5*dqkf(i);
      qkmb(i) = q_k(n,i) - 0.5*dqkb(i);
    }

    //--- Step 2. ------------------------------------------------------------------------
    // Calculate qi_ref (Mignone eq 41)
#pragma omp simd simdlen(SIMD_WIDTH)
    for (int i=il; i<=iu; ++i) {
      Real a1 = std::abs(q_k(n,i));
      Real a2 = std::abs(q_km1(n,i));
      Real a3 = std::abs(q_kp1(n,i));
      qkref(i) = cref_over_n*std::max(a1,std::max(a2, a3));
    }

    //--- Step 3. ------------------------------------------------------------------------
    // Calculate d, alpha, & omega values (Mignone eq 40, 42)
    Real dkp0 =  2.0*c1k(k); // c1k = wp1
    Real dkm0 = -2.0*c2k(k); // c2k = wm1

#pragma omp simd simdlen(SIMD_WIDTH)
    for (int i=il; i<=iu; ++i) {
      Real b0 = (1.0+SQR(dqkf(i)-dqkb(i))/(SQR(dqkf(i))+SQR(qkref(i))));
      Real b1 = (1.0+SQR(dqkf(i)-dqkb(i))/(SQR(dqkb(i))+SQR(qkref(i))));
      Real alpha_p0 = dkp0*b0;
      Real alpha_p1 = (1.0-dkp0)*b1;
      Real alpha_m0 = dkm0*b0;
      Real alpha_m1 = (1.0-dkm0)*b1;
      omega_p0(i) = alpha_p0/(alpha_p0+alpha_p1);
      omega_m0(i) = alpha_m0/(alpha_m0+alpha_m1);
    }

    //--- Step 4. ------------------------------------------------------------------------
    // Convert limited cell-centered values to interface-centered L/R Riemann states
    // both L/R values defined over [il,iu]
#pragma omp simd simdlen(SIMD_WIDTH)
    for (int i=il; i<=iu; ++i) {
      ql_kph(n,i) = omega_p0(i)*qkpf(i)+(1.0-omega_p0(i))*qkpb(i);
      qr_kmh(n,i) = omega_m0(i)*qkmf(i)+(1.0-omega_m0(i))*qkmb(i);
    }

#pragma omp simd simdlen(SIMD_WIDTH)
    for (int i=il; i<=iu; ++i) {
      Real dq2 = dqkf(i)*dqkb(i);
      Real dwm = 2.0*dq2/(dqkf(i)+dqkb(i));
      if (dq2 <= 0.0) dwm = 0.0;
      ql_plm(i) = q_k(n,i) + 0.5*dwm;
      qr_plm(i) = q_k(n,i) - 0.5*dwm;
    }

#pragma omp simd simdlen(SIMD_WIDTH)
    for (int i=il; i<=iu; ++i) {
      Real plcheck = 0.5*(1.0+SIGN(ql_kph(n,i)));
      Real prcheck = 0.5*(1.0+SIGN(qr_kmh(n,i)));
      ql_kph(n,i) = plcheck*ql_kph(n,i)+(1.0-plcheck)*ql_plm(i);
      qr_kmh(n,i) = prcheck*qr_kmh(n,i)+(1.0-prcheck)*qr_plm(i);
    }
  } // end char WENO3 loop over =nu

  // compute ql_(k+1/2) and qr_(k-1/2)
  for (int n=0; n<=nu; ++n) {
#pragma omp simd
    for (int i=il; i<=iu; ++i) {
      ql(n,i) = ql_kph(n,i);
      qr(n,i) = qr_kmh(n,i);
    }
  }
  return;
}
