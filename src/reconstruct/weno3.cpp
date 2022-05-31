//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file weno3.cpp
//! \brief WENO reconstruction (3rd order accuracy)
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
                             const AthenaArray<Real> &w, const AthenaArray<Real> &bcc,
                             AthenaArray<Real> &wl, AthenaArray<Real> &wr) {
  Coordinates *pco = pmy_block_->pcoord;

  // set work arrays used for primitive cell-averages to scratch
  AthenaArray<Real> &q_im1 = scr1_ni_, &q_i = scr2_ni_,
                    &q_ip1 = scr3_ni_, &qr_imh = scr4_ni_, &ql_iph = scr5_ni_,
                    &qr_plm = scr6_ni_, &ql_plm = scr7_ni_;

  // set work WENO3 arrays to shallow copies of scratch arrays:
  AthenaArray<Real> &dqif = scr01_i_, &dqib = scr02_i_, &qiref = scr03_i_,
                    &qipf = scr04_i_, &qipb = scr05_i_, qimf = scr06_i_, &qimb = scr07_i_,
                    &omega_p0 = scr08_i_, &omega_m0 = scr09_i_,
                    &plcheck = scr10_i_, &prcheck = scr11_i_,
                    &dip0 = scr12_i_, &dim0 = scr13_i_;

  // cache the x1-sliced primitive states for eigensystem calculation
  for (int n=0; n<NHYDRO; ++n) {
#pragma omp simd
    for (int i=il; i<=iu; ++i) {
      q_im1(n,i) = w(n,k,j,i-1);
      q_i  (n,i) = w(n,k,j,i  );
      q_ip1(n,i) = w(n,k,j,i+1);
    }
  }
  if (MAGNETIC_FIELDS_ENABLED) {
#pragma omp simd
    for (int i=il; i<=iu; ++i) {
      q_im1(IBY,i) = bcc(IB2,k,j,i-1);
      q_i  (IBY,i) = bcc(IB2,k,j,i  );
      q_ip1(IBY,i) = bcc(IB2,k,j,i+1);

      q_im1(IBZ,i) = bcc(IB3,k,j,i-1);
      q_i  (IBZ,i) = bcc(IB3,k,j,i  );
      q_ip1(IBZ,i) = bcc(IB3,k,j,i+1);
    }
  }

  // Calculate qiref using density profile (Mignone eq 41)
  const Real cref_over_n = 20.0/static_cast<Real>(pmy_block_->pmy_mesh->mesh_size.nx1);
#pragma omp simd
  for (int i=il; i<=iu; ++i) {
    Real a1 = std::abs(q_i  (IDN,i));
    Real a2 = std::abs(q_im1(IDN,i));
    Real a3 = std::abs(q_ip1(IDN,i));
    qiref(i) = cref_over_n*std::max(a1,std::max(a2, a3));
  }

  for (int n=0; n<NWAVE; ++n) {
    //--- Step 1. ------------------------------------------------------------------------
    // Calculate interface averages using PLM (Mignone eq 28, 29)
    if (uniform[X1DIR] && !curvilinear[X1DIR]) {
#pragma omp simd simdlen(SIMD_WIDTH)
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

    //--- Step 3. ------------------------------------------------------------------------
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
        ql_plm(n,i) = q_i(n,i) + 0.5*dwm;
        qr_plm(n,i) = q_i(n,i) - 0.5*dwm;
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
        ql_plm(n,i) = q_i(n,i) + ((pco->x1f(i+1) - pco->x1v(i))/pco->dx1f(i))*dwm;
        qr_plm(n,i) = q_i(n,i) - ((pco->x1v(i  ) - pco->x1f(i))/pco->dx1f(i))*dwm;
      }
    }
  } // end char WENO3 loop over NWAVE

  // cehck positivity of density and pressure
  if (NON_BAROTROPIC_EOS) {
#pragma omp simd simdlen(SIMD_WIDTH)
    for (int i=il; i<=iu; ++i) {
      plcheck(i) = 0.25*(1.0+SIGN(ql_iph(IDN,i)))
                   *(1.0+SIGN(ql_iph(IPR,i)));
      prcheck(i) = 0.25*(1.0+SIGN(qr_imh(IDN,i)))
                   *(1.0+SIGN(qr_imh(IPR,i)));
    }
  } else {
#pragma omp simd simdlen(SIMD_WIDTH)
    for (int i=il; i<=iu; ++i) {
      plcheck(i) = 0.5*(1.0+SIGN(ql_iph(IDN,i)));
      prcheck(i) = 0.5*(1.0+SIGN(qr_imh(IDN,i)));
    }
  }

  // compute ql_(i+1/2) and qr_(i-1/2)
  for (int n=0; n<NWAVE; ++n) {
#pragma omp simd
    for (int i=il; i<=iu; ++i) {
      wl(n,i+1) = plcheck(i)*ql_iph(n,i)+(1.0-plcheck(i))*ql_plm(n,i);
      wr(n,i  ) = prcheck(i)*qr_imh(n,i)+(1.0-prcheck(i))*qr_plm(n,i);
    }
  }
#pragma omp simd
  for (int i=il; i<=iu; ++i) {
    // Reapply EOS floors to both L/R reconstructed primitive states
    // TODO(felker): check that fused loop with NWAVE redundant application is slower
    pmy_block_->peos->ApplyPrimitiveFloors(wl, k, j, i+1);
    pmy_block_->peos->ApplyPrimitiveFloors(wr, k, j, i);
  }
  return;
}

//-------------------------------------------------------------------------------------
//! \fn Reconstruction::Weno3X2(const int k, const int j,
//!                             const int il, const int iu,
//!                             const AthenaArray<Real> &w, const AthenaArray<Real> &bcc,
//!                             AthenaArray<Real> &wl, AthenaArray<Real> &wr)
//! \brief Returns L/R interface values in X2-dir constructed using third-order WENO
//!        over [kl,ku][jl,ju][il,iu]

void Reconstruction::Weno3X2(const int k, const int j, const int il, const int iu,
                             const AthenaArray<Real> &w, const AthenaArray<Real> &bcc,
                             AthenaArray<Real> &wl, AthenaArray<Real> &wr) {
  Coordinates *pco = pmy_block_->pcoord;

  // set work arrays used for primitive cell-averages to scratch
  AthenaArray<Real> &q_jm1 = scr1_ni_, &q_j = scr2_ni_,
                    &q_jp1 = scr3_ni_, &qr_jmh = scr4_ni_, &ql_jph = scr5_ni_,
                    &qr_plm = scr6_ni_, &ql_plm = scr7_ni_;

  // set work WENO3 arrays to shallow copies of scratch arrays:
  AthenaArray<Real> &dqjf = scr01_i_, &dqjb = scr02_i_, &qjref = scr03_i_,
                    &qjpf = scr04_i_, &qjpb = scr05_i_, qjmf = scr06_i_, &qjmb = scr07_i_,
                    &omega_p0 = scr08_i_, &omega_m0 = scr09_i_,
                    &plcheck = scr10_i_, &prcheck = scr11_i_;

  // cache the x1-sliced primitive states for eigensystem calculation
  for (int n=0; n<NHYDRO; ++n) {
#pragma omp simd
    for (int i=il; i<=iu; ++i) {
      q_jm1(n,i) = w(n,k,j-1,i);
      q_j  (n,i) = w(n,k,j  ,i);
      q_jp1(n,i) = w(n,k,j+1,i);
    }
  }
  if (MAGNETIC_FIELDS_ENABLED) {
#pragma omp simd
    for (int i=il; i<=iu; ++i) {
      q_jm1(IBY,i) = bcc(IB3,k,j-1,i);
      q_j  (IBY,i) = bcc(IB3,k,j  ,i);
      q_jp1(IBY,i) = bcc(IB3,k,j+1,i);

      q_jm1(IBZ,i) = bcc(IB1,k,j-1,i);
      q_j  (IBZ,i) = bcc(IB1,k,j  ,i);
      q_jp1(IBZ,i) = bcc(IB1,k,j+1,i);
    }
  }

  // Calculate qiref using density profile (Mignone eq 41)
  const Real cref_over_n = 20.0/static_cast<Real>(pmy_block_->pmy_mesh->mesh_size.nx2);
#pragma omp simd
  for (int i=il; i<=iu; ++i) {
    Real a1 = std::abs(q_j  (IDN,i));
    Real a2 = std::abs(q_jm1(IDN,i));
    Real a3 = std::abs(q_jp1(IDN,i));
    qjref(i) = cref_over_n*std::max(a1,std::max(a2, a3));
  }

  for (int n=0; n<NWAVE; ++n) {
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

    //--- Step 3. ------------------------------------------------------------------------
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
        ql_plm(n,i) = q_j(n,i) + 0.5*dwm;
        qr_plm(n,i) = q_j(n,i) - 0.5*dwm;
      }
    } else {
#pragma omp simd simdlen(SIMD_WIDTH)
      for (int i=il; i<=iu; ++i) {
        Real dq2 = dqjf(i)*dqjb(i);
        Real cf = pco->dx2v(j  )/(pco->x2f(j+1) - pco->x2v(j));
        Real cb = pco->dx2v(j-1)/(pco->x2v(j  ) - pco->x2f(j));
        Real dwm = (dq2*(cf*dqjb(i)+cb*dqjf(i))/
                    (SQR(dqjb(i))+SQR(dqjf(i))+dq2*(cf+cb-2.0)));
        if (dq2 <= 0.0) dwm = 0.0;
        ql_plm(n,i) = q_j(n,i) + ((pco->x2f(j+1) - pco->x2v(j))/pco->dx2f(j))*dwm;
        qr_plm(n,i) = q_j(n,i) - ((pco->x2v(j  ) - pco->x2f(j))/pco->dx2f(j))*dwm;
      }
    }
  } // end char WENO3 loop over NWAVE

  // cehck positivity of density and pressure
  if (NON_BAROTROPIC_EOS) {
#pragma omp simd simdlen(SIMD_WIDTH)
    for (int i=il; i<=iu; ++i) {
      plcheck(i) = 0.25*(1.0+SIGN(ql_jph(IDN,i)))
                   *(1.0+SIGN(ql_jph(IPR,i)));
      prcheck(i) = 0.25*(1.0+SIGN(qr_jmh(IDN,i)))
                   *(1.0+SIGN(qr_jmh(IPR,i)));
    }
  } else {
#pragma omp simd simdlen(SIMD_WIDTH)
    for (int i=il; i<=iu; ++i) {
      plcheck(i) = 0.5*(1.0+SIGN(ql_jph(IDN,i)));
      prcheck(i) = 0.5*(1.0+SIGN(qr_jmh(IDN,i)));
    }
  }

  // check positivity of density and pressure
  for (int n=0; n<NWAVE; ++n) {
#pragma omp simd
    for (int i=il; i<=iu; ++i) {
      wl(n,i) = plcheck(i)*ql_jph(n,i)+(1.0-plcheck(i))*ql_plm(n,i);
      wr(n,i) = prcheck(i)*qr_jmh(n,i)+(1.0-prcheck(i))*qr_plm(n,i);
    }
  }

  // compute ql_(j+1/2) and qr_(j-1/2)
  for (int n=0; n<NWAVE; ++n) {
#pragma omp simd
    for (int i=il; i<=iu; ++i) {
      wl(n,i) = ql_jph(n,i);
      wr(n,i) = qr_jmh(n,i);
    }
  }
#pragma omp simd
  for (int i=il; i<=iu; ++i) {
    // Reapply EOS floors to both L/R reconstructed primitive states
    pmy_block_->peos->ApplyPrimitiveFloors(wl, k, j, i);
    pmy_block_->peos->ApplyPrimitiveFloors(wr, k, j, i);
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
                             const AthenaArray<Real> &w, const AthenaArray<Real> &bcc,
                             AthenaArray<Real> &wl, AthenaArray<Real> &wr) {
  Coordinates *pco = pmy_block_->pcoord;

  // set work arrays used for primitive cell-averages to scratch
  AthenaArray<Real> &q_km1 = scr1_ni_, &q_k = scr2_ni_,
                    &q_kp1 = scr3_ni_, &qr_kmh = scr4_ni_, &ql_kph = scr5_ni_,
                    &qr_plm = scr6_ni_, &ql_plm = scr7_ni_;

  // set work WENO arrays to shallow copies of scratch arrays:
  AthenaArray<Real> &dqkf = scr01_i_, &dqkb = scr02_i_, &qkref = scr03_i_,
                    &qkpf = scr04_i_, &qkpb = scr05_i_, qkmf = scr06_i_, &qkmb = scr07_i_,
                    &omega_p0 = scr08_i_, &omega_m0 = scr09_i_,
                    &plcheck = scr10_i_, &prcheck = scr11_i_;

  // cache the x1-sliced primitive states for eigensystem calculation
  for (int n=0; n<NHYDRO; ++n) {
#pragma omp simd
    for (int i=il; i<=iu; ++i) {
      q_km1(n,i) = w(n,k-1,j,i);
      q_k  (n,i) = w(n,k  ,j,i);
      q_kp1(n,i) = w(n,k+1,j,i);
    }
  }
  if (MAGNETIC_FIELDS_ENABLED) {
#pragma omp simd
    for (int i=il; i<=iu; ++i) {
      q_km1(IBY,i) = bcc(IB1,k-1,j,i);
      q_k  (IBY,i) = bcc(IB1,k  ,j,i);
      q_kp1(IBY,i) = bcc(IB1,k+1,j,i);

      q_km1(IBZ,i) = bcc(IB2,k-1,j,i);
      q_k  (IBZ,i) = bcc(IB2,k  ,j,i);
      q_kp1(IBZ,i) = bcc(IB2,k+1,j,i);
    }
  }

  // Calculate qiref using density profile (Mignone eq 41)
  const Real cref_over_n = 20.0/static_cast<Real>(pmy_block_->pmy_mesh->mesh_size.nx3);
#pragma omp simd
  for (int i=il; i<=iu; ++i) {
    Real a1 = std::abs(q_k  (IDN,i));
    Real a2 = std::abs(q_km1(IDN,i));
    Real a3 = std::abs(q_kp1(IDN,i));
    qkref(i) = cref_over_n*std::max(a1,std::max(a2, a3));
  }

  for (int n=0; n<NWAVE; ++n) {
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

    //--- Step 3. ------------------------------------------------------------------------
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
      ql_plm(n,i) = q_k(n,i) + 0.5*dwm;
      qr_plm(n,i) = q_k(n,i) - 0.5*dwm;
    }
  } // end char PPM loop over NWAVE

  // cehck positivity of density and pressure
  if (NON_BAROTROPIC_EOS) {
#pragma omp simd simdlen(SIMD_WIDTH)
    for (int i=il; i<=iu; ++i) {
      plcheck(i) = 0.25*(1.0+SIGN(ql_kph(IDN,i)))
                   *(1.0+SIGN(ql_kph(IPR,i)));
      prcheck(i) = 0.25*(1.0+SIGN(qr_kmh(IDN,i)))
                   *(1.0+SIGN(qr_kmh(IPR,i)));
    }
  } else {
#pragma omp simd simdlen(SIMD_WIDTH)
    for (int i=il; i<=iu; ++i) {
      plcheck(i) = 0.5*(1.0+SIGN(ql_kph(IDN,i)));
      prcheck(i) = 0.5*(1.0+SIGN(qr_kmh(IDN,i)));
    }
  }

  // compute ql_(k+1/2) and qr_(k-1/2)
  for (int n=0; n<NWAVE; ++n) {
#pragma omp simd
    for (int i=il; i<=iu; ++i) {
      wl(n,i) = plcheck(i)*ql_kph(n,i)+(1.0-plcheck(i))*ql_plm(n,i);
      wr(n,i) = prcheck(i)*qr_kmh(n,i)+(1.0-prcheck(i))*qr_plm(n,i);
    }
  }
#pragma omp simd
  for (int i=il; i<=iu; ++i) {
    // Reapply EOS floors to both L/R reconstructed primitive states
    pmy_block_->peos->ApplyPrimitiveFloors(wl, k, j, i);
    pmy_block_->peos->ApplyPrimitiveFloors(wr, k, j, i);
  }
  return;
}
