/*
//@HEADER
// ***********************************************************************
//
//       Ifpack2: Templated Object-Oriented Algebraic Preconditioner Package
//                 Copyright (2009) Sandia Corporation
//
// Under terms of Contract DE-AC04-94AL85000, there is a non-exclusive
// license for use of this work by or on behalf of the U.S. Government.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// ***********************************************************************
//@HEADER
*/

#ifndef IFPACK2_DETAILS_SSTEPCHEBYSHEVKERNEL_DEF_HPP
#define IFPACK2_DETAILS_SSTEPCHEBYSHEVKERNEL_DEF_HPP

#include "Tpetra_CrsMatrix.hpp"
#include "Tpetra_MultiVector.hpp"
#include "Tpetra_Operator.hpp"
#include "Tpetra_Vector.hpp"
#include "Tpetra_withLocalAccess_MultiVector.hpp"
#ifdef OLD_APPROACH_COMMUNICATION_NEEDED
#include "Tpetra_Export_decl.hpp"
#include "Tpetra_Import_decl.hpp"
#endif //OLD_APPROACH_COMMUNICATION_NEEDED
#include "Kokkos_ArithTraits.hpp"
#include "Teuchos_Assert.hpp"
#include <type_traits>

//#define RANKPRINT rank==1
#define RANKPRINT rank==666

namespace Ifpack2 {
namespace Details {
namespace Impl {

/// \brief Functor for computing W := alpha * D * (B - A*X) + beta * W and X := X+W.
///
/// This is an implementation detail of s-step chebyshev_kernel_vector,
/// which in turn is an implementation detail of sStepChebyshevKernel.
template<class WVector,
         class DVector,
         class BVector,
         class AMatrix,
         class XVector,
         class Scalar,
         bool use_beta>
struct sStepChebyshevKernelVectorFunctor {
  static_assert (static_cast<int> (WVector::Rank) == 1,
                 "WVector must be a rank 1 View.");
  static_assert (static_cast<int> (DVector::Rank) == 1,
                 "DVector must be a rank 1 View.");
  static_assert (static_cast<int> (BVector::Rank) == 1,
                 "BVector must be a rank 1 View.");
  static_assert (static_cast<int> (XVector::Rank) == 1,
                 "XVector must be a rank 1 View.");

  using execution_space = typename AMatrix::execution_space;
  using LO = typename AMatrix::non_const_ordinal_type;
  using value_type = typename AMatrix::non_const_value_type;
  using team_policy = typename Kokkos::TeamPolicy<execution_space>;
  using team_member = typename team_policy::member_type;
  using ATV = Kokkos::ArithTraits<value_type>;

  const Scalar alpha;
  WVector m_w;
  DVector m_d;
  BVector m_b;
  AMatrix m_A;
  XVector m_x;
  const Scalar beta;

  const LO rows_per_team;
  const int rank;
  const int numLocalRows;

  sStepChebyshevKernelVectorFunctor (const Scalar& alpha_,
                                     const WVector& m_w_,
                                     const DVector& m_d_,
                                     const BVector& m_b_,
                                     const AMatrix& m_A_,
                                     const XVector& m_x_,
                                     const Scalar& beta_,
                                     const int rows_per_team_,
                                     const int rank,
                                     const int numLocalRows) :
    alpha (alpha_),
    m_w (m_w_),
    m_d (m_d_),
    m_b (m_b_),
    m_A (m_A_),
    m_x (m_x_),
    beta (beta_),
    rows_per_team (rows_per_team_),
    rank(rank),
    numLocalRows(numLocalRows)
  {
    const size_t numRows = m_A.numRows ();
    const size_t numCols = m_A.numCols ();

    const size_t m_d_extent = size_t(m_d.extent(0));
    const size_t m_b_extent = size_t(m_b.extent(0));
    const size_t m_w_extent = size_t(m_w.extent(0));
    const size_t m_x_extent = size_t(m_x.extent(0));
    if (RANKPRINT) {
      std::cout << "(" << rank << ") m_d=" << m_d_extent << ",m_b="
                << m_b_extent << ",m_w="
                << m_w_extent << ",m_x="
                << m_x_extent << ",numRows="
                << numRows << ",numCols="
                << numCols << ",numLocalRows="
                << numLocalRows << std::endl;
    }

    TEUCHOS_ASSERT( m_w.extent (0) == m_d.extent (0) );
    TEUCHOS_ASSERT( m_w.extent (0) == m_b.extent (0) );
    //TEUCHOS_ASSERT( numRows == size_t (m_w.extent (0)) );
    //TEUCHOS_ASSERT( numCols <= size_t (m_x.extent (0)) );
  }

  KOKKOS_INLINE_FUNCTION
  void operator() (const team_member& dev) const
  {
    using residual_value_type = typename BVector::non_const_value_type;
    using KAT = Kokkos::ArithTraits<residual_value_type>;

    Kokkos::parallel_for
      (Kokkos::TeamThreadRange (dev, 0, rows_per_team),
       [&] (const LO& loop) {
         const LO lclRow =
           static_cast<LO> (dev.league_rank ()) * rows_per_team + loop;
         if (lclRow >= numLocalRows) {
           return;
         }
         if (RANKPRINT) {
           printf("  [%d] lclRow=%d, dev.league_rank=%d, rows_per_team=%d, loop=%d\n",rank,lclRow,
                  dev.league_rank (), rows_per_team, loop);
           fflush(stdout);
         }
         const auto A_row = m_A.rowConst(lclRow);
         const LO row_length = static_cast<LO> (A_row.length);
         residual_value_type A_x = KAT::zero ();

         Kokkos::parallel_reduce (Kokkos::ThreadVectorRange (dev, row_length), [&] (const LO iEntry, residual_value_type& lsum) {
              const auto A_val = A_row.value(iEntry);
              //printf("iEntry=%d\n",iEntry);
              if (RANKPRINT) {
                printf("  [%d] A_row.colidx(%d)=%d\n",rank, iEntry,A_row.colidx(iEntry));
                fflush(stdout);
              }
              lsum += A_val * m_x(A_row.colidx(iEntry));
            }, A_x);

         Kokkos::single
           (Kokkos::PerThread(dev),
            [&] () {
              if (RANKPRINT) {
                printf("  [%d] lclRow=%d, m_d.extend(0)=%lu, m_b.extent(0)=%lu, A.numRows=%d, m_w.extent()=%lu, m_x.extent()=%lu\n",rank,lclRow,m_d.extent(0),m_b.extent(0),m_A.numRows(),m_w.extent(0),m_x.extent(0));
                fflush(stdout);
              }
              //printf("[%d] DO NOTHING lclRow=%d, m_d.extend(0)=%d, m_b.extent(0)=%d\n",rank,lclRow,m_d.extent(0),m_b.extent(0));
#define SKIPFORNOW
#ifdef SKIPFORNOW
              const auto alpha_D_res =
                alpha * m_d(lclRow) * (m_b(lclRow) - A_x);
              if (use_beta) {
                m_w(lclRow) = beta * m_w(lclRow) + alpha_D_res;
              }
              else {
                m_w(lclRow) = alpha_D_res;
              }
#endif
            });
       });
  }
}; //sStepChebyshevKernelVectorFunctor

template<class ExecutionSpace>
int64_t
sstep_chebyshev_kernel_vector_launch_parameters (int64_t numRows,
                                                 int64_t nnz,
                                                 int64_t rows_per_thread,
                                                 int& team_size,
                                                 int& vector_length)
{
  using execution_space = typename ExecutionSpace::execution_space;

  int64_t rows_per_team;
  int64_t nnz_per_row = nnz/numRows;

  if (nnz_per_row < 1) {
    nnz_per_row = 1;
  }

  if (vector_length < 1) {
    vector_length = 1;
    while (vector_length<32 && vector_length*6 < nnz_per_row) {
      vector_length *= 2;
    }
  }

  // Determine rows per thread
  if (rows_per_thread < 1) {
#ifdef KOKKOS_ENABLE_CUDA
    if (std::is_same<Kokkos::Cuda, execution_space>::value) {
      rows_per_thread = 1;
    }
    else
#endif
    {
      if (nnz_per_row < 20 && nnz > 5000000) {
        rows_per_thread = 256;
      }
      else {
        rows_per_thread = 64;
      }
    }
  }

#ifdef KOKKOS_ENABLE_CUDA
  if (team_size < 1) {
    if (std::is_same<Kokkos::Cuda,execution_space>::value) {
      team_size = 256/vector_length;
    }
    else {
      team_size = 1;
    }
  }
#endif

  rows_per_team = rows_per_thread * team_size;

  if (rows_per_team < 0) {
    int64_t nnz_per_team = 4096;
    int64_t conc = execution_space::concurrency ();
    while ((conc * nnz_per_team * 4 > nnz) &&
           (nnz_per_team > 256)) {
      nnz_per_team /= 2;
    }
    rows_per_team = (nnz_per_team + nnz_per_row - 1) / nnz_per_row;
  }

  return rows_per_team;
} //sstep_chebyshev_kernel_vector_launch_parameters

// W := alpha * D * (B - A*X) + beta * W.
template<class WVector,
         class DVector,
         class BVector,
         class AMatrix,
         class XVector,
         class Scalar>
static void
sstep_chebyshev_kernel_vector
(const Scalar& alpha,
 const WVector& w,
 const DVector& d,
 const BVector& b,
 const AMatrix& A,
 const XVector& x,
 const Scalar& beta,
 const int rank, 
 const int numLocalRows)
{
  using execution_space = typename AMatrix::execution_space;

  if (A.numRows () == 0) {
    return;
  }

  int team_size = -1;
  int vector_length = -1;
  int64_t rows_per_thread = -1;

  const int64_t rows_per_team =
    sstep_chebyshev_kernel_vector_launch_parameters<execution_space>
      //(A.numRows (), A.nnz (), rows_per_thread, team_size, vector_length);
      (numLocalRows, A.nnz (), rows_per_thread, team_size, vector_length);
  int64_t worksets = (b.extent (0) + rows_per_team - 1) / rows_per_team;

  using Kokkos::Dynamic;
  using Kokkos::Schedule;
  using Kokkos::TeamPolicy;
  using policy_type = TeamPolicy<execution_space, Schedule<Dynamic>>;
  const char kernel_label[] = "sstep_chebyshev_kernel_vector";
  policy_type policy (1, 1);
  if (team_size < 0) {
    policy = policy_type (worksets, Kokkos::AUTO, vector_length);
  }
  else {
    policy = policy_type (worksets, team_size, vector_length);
  }

  // Canonicalize template arguments to avoid redundant instantiations.
  using w_vec_type = typename WVector::non_const_type;
  using d_vec_type = typename DVector::const_type;
  using b_vec_type = typename BVector::const_type;
  using matrix_type = AMatrix;
  using x_vec_type = typename XVector::const_type;
  using scalar_type = typename Kokkos::ArithTraits<Scalar>::val_type;

  if (beta == Kokkos::ArithTraits<Scalar>::zero ()) {
    constexpr bool use_beta = false;
    using functor_type =
      sStepChebyshevKernelVectorFunctor<w_vec_type, d_vec_type,
                                        b_vec_type, matrix_type,
                                        x_vec_type, scalar_type,
                                        use_beta>;
    functor_type func (alpha, w, d, b, A, x, beta, rows_per_team, rank, numLocalRows);
    Kokkos::parallel_for (kernel_label, policy, func);
  }
  else {
    constexpr bool use_beta = true;
    using functor_type =
      sStepChebyshevKernelVectorFunctor<w_vec_type, d_vec_type,
                                        b_vec_type, matrix_type,
                                        x_vec_type, scalar_type,
                                        use_beta>;
    functor_type func (alpha, w, d, b, A, x, beta, rows_per_team, rank, numLocalRows);
    Kokkos::parallel_for (kernel_label, policy, func);
  }
} //sstep_chebyshev_kernel_vector

} // namespace Impl

template<class TpetraOperatorType>
sStepChebyshevKernel<TpetraOperatorType>::
sStepChebyshevKernel (
                      const Teuchos::RCP<const operator_type>& A,
                      const Teuchos::RCP<const crs_matrix_type>& locA,
                      const Teuchos::RCP<const crs_matrix_type>& extA)
{
  setMatrix (A,locA,extA);
}

template<class TpetraOperatorType>
void
sStepChebyshevKernel<TpetraOperatorType>::
setMatrix (
           const Teuchos::RCP<const operator_type>& A,
           const Teuchos::RCP<const crs_matrix_type>& locA,
           const Teuchos::RCP<const crs_matrix_type>& extA)
{
  A_op_ = A;
  locA_ = locA;
  extA_ = extA;
}

#ifdef OLD_APPROACH_COMMUNICATION_NEEDED
template<class TpetraOperatorType>
void
sStepChebyshevKernel<TpetraOperatorType>::
setMatrix (const Teuchos::RCP<const operator_type>& A)
{
  if (A_op_.get () != A.get ()) {
    A_op_ = A;

    // We'll (re)allocate these on demand.
    X_colMap_ = std::unique_ptr<vector_type> (nullptr);
    V1_ = std::unique_ptr<multivector_type> (nullptr);

    using Teuchos::rcp_dynamic_cast;
    Teuchos::RCP<const crs_matrix_type> A_crs =
      rcp_dynamic_cast<const crs_matrix_type> (A);
    if (A_crs.is_null ()) {
      A_crs_ = Teuchos::null;
      imp_ = Teuchos::null;
      exp_ = Teuchos::null;
    }
    else {
      TEUCHOS_ASSERT( A_crs->isFillComplete () );
      A_crs_ = A_crs;
      auto G = A_crs->getCrsGraph ();
      imp_ = G->getImporter ();
      exp_ = G->getExporter ();
    }
  }
}
#endif //OLD_APPROACH_COMMUNICATION_NEEDED

template<class TpetraOperatorType>
void
sStepChebyshevKernel<TpetraOperatorType>::
apply (multivector_type& W,
         const SC& alpha,
         vector_type& D_inv,
         multivector_type& B,
         multivector_type& X,
         const SC& beta,
         Teuchos::ArrayView<const size_t> &hstarts, const int &hind, const int &rank)
         //const size_t &halo_start)
{
  using Teuchos::RCP;
  using Teuchos::rcp;

  printf("HEREEEEEEEEEEEEEEEEEEEEEEEE\n");
  fflush(stdout);

  if (canFuse (B)) {
    // "nonconst" here has no effect other than on the return type.
    if (W_vec_.is_null() || W.getLocalViewHost().data() != viewW_.data()) {
      viewW_ = W.getLocalViewHost();
      W_vec_ = W.getVectorNonConst (0);
    }
    if (B_vec_.is_null() || B.getLocalViewHost().data() != viewB_.data()) {
      viewB_ = B.getLocalViewHost();
      B_vec_ = B.getVectorNonConst (0);
    }
    if (X_vec_.is_null() || X.getLocalViewHost().data() != viewX_.data()) {
      viewX_ = X.getLocalViewHost();
      X_vec_ = X.getVectorNonConst (0);
    }
    //TEUCHOS_ASSERT( ! A_crs_.is_null () );
    TEUCHOS_ASSERT( ! locA_.is_null () );
    TEUCHOS_ASSERT( ! extA_.is_null () );
    //fusedCase (*W_vec_, alpha, D_inv, *B_vec_, *locA_, *extA_, *X_vec_, beta, halo_start);
    fusedCase (*W_vec_, alpha, D_inv, *B_vec_, *locA_, *extA_, *X_vec_, beta, hstarts, hind, rank);
  }
  else {
    TEUCHOS_ASSERT( ! A_op_.is_null () );
    unfusedCase (W, alpha, D_inv, B, *A_op_, *X_vec_, beta);
  }
} //apply

#ifdef OLD_APPROACH_COMMUNICATION_NEEDED
template<class TpetraOperatorType>
typename sStepChebyshevKernel<TpetraOperatorType>::vector_type&
sStepChebyshevKernel<TpetraOperatorType>::
importVector (vector_type& X_domMap)
{
  if (imp_.is_null ()) {
    return X_domMap;
  }
  else {
    if (X_colMap_.get () == nullptr) {
      using V = vector_type;
      X_colMap_ = std::unique_ptr<V> (new V (imp_->getTargetMap ()));
    }
    X_colMap_->doImport (X_domMap, *imp_, Tpetra::REPLACE);
    return *X_colMap_;
  }
} //importVector
#endif //OLD_APPROACH_COMMUNICATION_NEEDED

template<class TpetraOperatorType>
bool
sStepChebyshevKernel<TpetraOperatorType>::
canFuse (const multivector_type& B) const
{
//  return B.getNumVectors () == size_t (1) &&
//    ! A_crs_.is_null () &&
//    exp_.is_null ();
  return B.getNumVectors () == size_t (1) &&
    ! locA_.is_null () &&
    exp_.is_null ();
}

template<class TpetraOperatorType>
void
sStepChebyshevKernel<TpetraOperatorType>::
unfusedCase (multivector_type& W,
             const SC& alpha,
             vector_type& D_inv,
             multivector_type& B,
             const operator_type& A,
             multivector_type& X,
             const SC& beta)
{
  using STS = Teuchos::ScalarTraits<SC>;
  if (V1_.get () == nullptr) {
    using MV = multivector_type;
    const size_t numVecs = B.getNumVectors ();
    V1_ = std::unique_ptr<MV> (new MV (B.getMap (), numVecs));
  }
  const SC one = Teuchos::ScalarTraits<SC>::one ();

  // V1 = B - A*X
  Tpetra::deep_copy (*V1_, B);
  A.apply (X, *V1_, Teuchos::NO_TRANS, -one, one);

  // W := alpha * D_inv * V1 + beta * W
  W.elementWiseMultiply (alpha, D_inv, *V1_, beta);

} //sStepChebyshevKernel::unfusedCase

template<class TpetraOperatorType>
void
sStepChebyshevKernel<TpetraOperatorType>::
fusedCase (vector_type& W,
           const SC& alpha,
           vector_type& D_inv,
           vector_type& B,
           const crs_matrix_type& locA,
           const crs_matrix_type& extA,
           vector_type& X,
           const SC& beta,
           Teuchos::ArrayView<const size_t> &hstarts,
           const int &hind,
           const int &rank)
{

#if 0 //INCORRECT_LOCAL_VIEWS
  //FIXME do halo stuff here
  int yrange = hstarts[hind];
  auto B_lcl = B.getLocalViewDevice ();
  auto W_lcl = W.getLocalViewDevice ();
  auto X_lcl = X.getLocalViewDevice ();
  auto Dinv_lcl = D_inv.getLocalViewDevice ();
  auto numLocalRows = locA.getNodeNumRows();

  auto W_ext = Kokkos::subview(W_lcl,std::make_pair(numLocalRows,numLocalRows+yrange),Kokkos::ALL());
  auto B_ext = Kokkos::subview(B_lcl,std::make_pair(numLocalRows,numLocalRows+yrange),Kokkos::ALL());
  auto Dinv_ext = Kokkos::subview(Dinv_lcl,std::make_pair(numLocalRows,numLocalRows+yrange),Kokkos::ALL());
  int xlimit = ( (hind == hstarts.size()-1) ? X_lcl.extent(0) : numLocalRows+hstarts[hind+1] );
  auto X_ext = Kokkos::subview(X_lcl,std::make_pair(0,xlimit),Kokkos::ALL());
#endif

  //auto numLocalRows = hstarts[hind];
  auto numLocalRows = locA.getNodeNumRows();
  int yrange = hstarts[hind];

  //TODO JHU working in here 6-Mar-2020
  // W = B - A*X

  // Only need these aliases because we lack C++14 generic lambdas.
  using Tpetra::with_local_access_function_argument_type;
  using ro_lcl_vec_type = with_local_access_function_argument_type< decltype (readOnly (B))>;
  using wo_lcl_vec_type = with_local_access_function_argument_type< decltype (writeOnly (B))>;
  using rw_lcl_vec_type = with_local_access_function_argument_type< decltype (readWrite (B))>;

  using Tpetra::withLocalAccess;
  using Tpetra::readOnly;
  using Tpetra::readWrite;
  using Tpetra::writeOnly;
  using Impl::sstep_chebyshev_kernel_vector;
  using STS = Teuchos::ScalarTraits<SC>;

  auto A_lcl = locA.getLocalMatrix ();
  auto A_ext = extA.getLocalMatrix ();
  if (beta == STS::zero ()) {

    if (RANKPRINT) {
      printf("(beta=0) [%d] local matvec\n", rank);
      fflush(stdout);
    }
    // matrix, local part
    withLocalAccess
      ([&] (const wo_lcl_vec_type& W_lcl,
            const ro_lcl_vec_type& D_inv_lcl,
            const ro_lcl_vec_type& B_lcl,
            const ro_lcl_vec_type& X_lcl) {
         sstep_chebyshev_kernel_vector (alpha, W_lcl, D_inv_lcl,
                                        B_lcl, A_lcl, X_lcl, beta, rank, A_lcl.numRows());
       },
       writeOnly (W),
       readOnly (D_inv),
       readOnly (B),
       readOnly (X)); 
       //writeOnly (X));

    // matrix, halo part
    if (hind>0) //overlap > 0
    withLocalAccess
      ([&] (const wo_lcl_vec_type& W_lcl,
            const ro_lcl_vec_type& D_inv_lcl,
            const ro_lcl_vec_type& B_lcl,
            const wo_lcl_vec_type& X_lcl) {
         //maybe doesn't like auto?
         //auto W_ext = Kokkos::subview(W_lcl,std::make_pair(numLocalRows,numLocalRows+yrange),Kokkos::ALL());
    if (RANKPRINT) {
      printf("Halo Starts:");
      for(size_t i=0; i< (size_t)hstarts.size(); i++)
        printf("%d ",(int) hstarts[i]);
      printf("\n");
      fflush(stdout);
    }
         auto W_ext = Kokkos::subview(W_lcl,std::make_pair(numLocalRows,numLocalRows+yrange));
         auto B_ext = Kokkos::subview(B_lcl,std::make_pair(numLocalRows,numLocalRows+yrange));
         auto Dinv_ext = Kokkos::subview(D_inv_lcl,std::make_pair(numLocalRows,numLocalRows+yrange));
         int xlimit = ( (hind == hstarts.size()-1) ? X_lcl.extent(0) : numLocalRows+hstarts[hind+1] );
         auto X_ext = Kokkos::subview(X_lcl,std::make_pair(0,xlimit));
  if (RANKPRINT) {
    printf("(beta=0) [%d] overlapped matvec\n", rank);
    printf("(beta=0) [%d] numLocalRows=%lu, hind=%d, yrange=%d, xlimit=%d, A_ext.numRows=%d, hstarts[hind+1]=%lu\n", rank, numLocalRows, hind, yrange,xlimit,A_ext.numRows(),hstarts[hind+1]);
    fflush(stdout);
  }
         sstep_chebyshev_kernel_vector (alpha, W_ext, Dinv_ext,
                                        B_ext, A_ext, X_ext, beta, rank, yrange);
                                        //B_ext, A_ext, X_ext, beta, rank, numLocalRows);
       },
       writeOnly (W),
       readOnly (D_inv),
       readOnly (B),
       writeOnly (X));
  }
  else { // need to read _and_ write W if beta != 0
/*
    withLocalAccess
      ([&] (const rw_lcl_vec_type& W_lcl,
            const ro_lcl_vec_type& D_lcl,
            const ro_lcl_vec_type& B_lcl,
            const ro_lcl_vec_type& X_lcl) {
         sstep_chebyshev_kernel_vector (alpha, W_lcl, D_lcl,
                                        B_lcl, A_lcl, X_lcl, beta);
       },
       readWrite (W),
       readOnly (D_inv),
       readOnly (B),
       readOnly (X));
*/
    // matrix, local part
    if (RANKPRINT) {
      printf("(beta!=0) [%d] local matvec\n", rank);
      fflush(stdout);
    }
    withLocalAccess
      ([&] (const rw_lcl_vec_type& W_lcl,
            const ro_lcl_vec_type& D_inv_lcl,
            const ro_lcl_vec_type& B_lcl,
            const ro_lcl_vec_type& X_lcl) {
         sstep_chebyshev_kernel_vector (alpha, W_lcl, D_inv_lcl,
                                        B_lcl, A_lcl, X_lcl, beta, rank, A_lcl.numRows());
       },
       readWrite (W),
       readOnly (D_inv),
       readOnly (B),
       readOnly (X)); 
       //writeOnly (X));

    // matrix, halo part
    if (RANKPRINT) {
      printf("(beta!=0) [%d] overlapped matvec\n", rank);
      fflush(stdout);
    }
    if (hind>0)
    withLocalAccess
      ([&] (const rw_lcl_vec_type& W_lcl,
            const ro_lcl_vec_type& D_inv_lcl,
            const ro_lcl_vec_type& B_lcl,
            const wo_lcl_vec_type& X_lcl) {
         auto W_ext = Kokkos::subview(W_lcl,std::make_pair(numLocalRows,numLocalRows+yrange));
         auto B_ext = Kokkos::subview(B_lcl,std::make_pair(numLocalRows,numLocalRows+yrange));
         auto Dinv_ext = Kokkos::subview(D_inv_lcl,std::make_pair(numLocalRows,numLocalRows+yrange));
         int xlimit = ( (hind == hstarts.size()-1) ? X_lcl.extent(0) : numLocalRows+hstarts[hind+1] );
         auto X_ext = Kokkos::subview(X_lcl,std::make_pair(0,xlimit));
         if (RANKPRINT) {
           printf("(beta!=0) [%d] numLocalRows=%lu, hind=%d, yrange=%d, xlimit=%d, A_ext.numRows=%d\n", rank, numLocalRows, hind, yrange,xlimit,A_ext.numRows());
           fflush(stdout);
         }
         sstep_chebyshev_kernel_vector (alpha, W_ext, Dinv_ext,
                                        B_ext, A_ext, X_ext, beta, rank, yrange);
                                        //B_ext, A_ext, X_ext, beta, rank, numLocalRows);
       },
       readWrite (W),
       readOnly (D_inv),
       readOnly (B),
       writeOnly (X));
  }
} //sStepChebyshevKernel::fusedCase

} // namespace Details
} // namespace Ifpack2

#define IFPACK2_DETAILS_SSTEPCHEBYSHEVKERNEL_INSTANT(SC,LO,GO,NT) \
  template class Ifpack2::Details::sStepChebyshevKernel<Tpetra::Operator<SC, LO, GO, NT> >;

#endif // IFPACK2_DETAILS_SSTEPCHEBYSHEVKERNEL_DEF_HPP
