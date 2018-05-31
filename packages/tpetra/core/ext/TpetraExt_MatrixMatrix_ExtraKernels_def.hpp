// @HEADER
// ***********************************************************************
//
//          Tpetra: Templated Linear Algebra Services Package
//                 Copyright (2008) Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software.
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
// Questions? Contact Michael A. Heroux (maherou@sandia.gov)
//
// ************************************************************************
// @HEADER

#ifndef TPETRA_MATRIXMATRIX_EXTRAKERNELS_DEF_HPP
#define TPETRA_MATRIXMATRIX_EXTRAKERNELS_DEF_HPP
#include "TpetraExt_MatrixMatrix_ExtraKernels_decl.hpp"

#define HAVE_TPETRA_INST_OPENMP
#include <omp.h>
#include <cstddef>

extern int WARN_ON_REALLOC;

#if defined (HAVE_TPETRA_INST_OPENMP)

#define ENABLE_NESTED 0
#define ENABLE_USE_OMP_BARRIER 0
#define ENABLE_BLOCKED_COLS 0
#define ENABLE_FORCE_ATOMIC_ADD 0

#define ALLOC_PAGE_ALIGNED 1
#define ENABLE_HUGEPAGE_ALLOC 1
#define DEBUG_ALLOC 0

#if ENABLE_HUGEPAGE_ALLOC == 1
#define MY_PAGE_SIZE_BYTES size_t(2*1024*1024)
#else
#define MY_PAGE_SIZE_BYTES size_t(4*1024)
#endif

#define ENABLE_WARN_ON_REALLOC 1
#define ENABLE_RAW_POINTERS 0
#define ENABLE_RESTRICT 0

#define STR_HELPER(x) #x
#define STR(x) STR_HELPER(x)

#pragma message "ENABLE_NESTED = " STR(ENABLE_NESTED)
#pragma message "ENABLE_USE_OMP_BARRIER = " STR(ENABLE_USE_OMP_BARRIER)
#pragma message "ENABLE_BLOCKED_COLS = " STR(ENABLE_BLOCKED_COLS)
#pragma message "ENABLE_FORCE_ATOMIC_ADD = " STR(ENABLE_FORCE_ATOMIC_ADD)

#pragma message "ALLOC_PAGE_ALIGNED = " STR(ALLOC_PAGE_ALIGNED)
#pragma message "ENABLE_HUGEPAGE_ALLOC = " STR(ENABLE_HUGEPAGE_ALLOC)
#pragma message "DEBUG_ALLOC = " STR(DEBUG_ALLOC)

#pragma message "MY_PAGE_SIZE_BYTES = " STR(MY_PAGE_SIZE_BYTES)
#pragma message "ENABLE_WARN_ON_REALLOC = " STR(ENABLE_WARN_ON_REALLOC)
#pragma message "ENABLE_RAW_POINTERS = " STR(ENABLE_RAW_POINTERS)
#pragma message "ENABLE_RESTRICT = " STR(ENABLE_RESTRICT)

#if ENABLE_RESTRICT == 1
#ifdef __INTEL_COMPILER
#define __restrict restrict
#endif
#else
#define __restrict
#endif

namespace {

template <typename View>
using UnmanagedView = Kokkos::View< typename View::data_type
                                  , typename View::array_layout
                                  , typename View::device_type
                                  , typename Kokkos::MemoryTraits< Kokkos::Unmanaged>
                                   >;

static inline int fetch_and_add(volatile int* variable, int value)
{
  __asm__ volatile("lock; xaddl %0, %1"
                   : "+r" (value), "+m" (*variable) // input+output
                   : // No input-only
                   : "memory"
                   );
  return value;
}

// Grow memory to atleast minimum_count
// if *ptr == NULL, then this allocates memory (not realloc)
static
size_t grow_ptr (const size_t minimum_count,
                 void ** ptr,
                 const size_t scalar_size,
                 const bool fill_to_pagesize = false) {
  size_t new_sz;
  size_t new_count;

  if (fill_to_pagesize) {
    const size_t num_pages = (scalar_size * minimum_count) / MY_PAGE_SIZE_BYTES
                             +
                             (((scalar_size * minimum_count) % MY_PAGE_SIZE_BYTES) == 0 ? 0 : 1);

    new_sz = num_pages * MY_PAGE_SIZE_BYTES;
    new_count = new_sz / scalar_size;
  } else {
    new_count = minimum_count;
    new_sz = new_count * scalar_size;
  }

  if ((*ptr) == nullptr) {
    #if ALLOC_PAGE_ALIGNED == 1
    auto rc = posix_memalign(ptr, MY_PAGE_SIZE_BYTES, new_sz); // rc == 0 or DIE

      #if DEBUG_ALLOC == 1
      {
        std::stringstream ss;
        ss << "Thread[" << omp_get_thread_num() << "] posix_memalign! "
           << "rc = " << rc << ", PageSize = " << sysconf(_SC_PAGESIZE) << ", Requested Alignment = " << MY_PAGE_SIZE_BYTES
           << ", Scalar Size "
           << scalar_size << ", requested_count = " << minimum_count << ", NEW count = " << new_count
                                                               << ", NEW sz = " << new_sz << std::endl
           << "Ptr: " << ptr
           << ", Alligned: " << std::boolalpha << (((uintptr_t)(*ptr) & (uintptr_t)(MY_PAGE_SIZE_BYTES - 1)) == 0)
           << std::endl;
        std::cerr << ss.str ();
      }
      #endif

    #else
      *ptr = malloc (new_sz);

      #if DEBUG_ALLOC == 1
      {
        std::stringstream ss;
        ss << "Thread[" << omp_get_thread_num() << "] malloc! "
           << "rc = " << ptr << ", PageSize = " << sysconf(_SC_PAGESIZE)
           << ", Scalar Size: "
           << scalar_size << ", requested_count = " << minimum_count << ", NEW count = " << new_count
                                                                     << ", NEW sz = " << new_sz << std::endl
           << "Ptr: " << ptr
           << ", Alligned: " << std::boolalpha << (((uintptr_t)(*ptr) & (uintptr_t)(MY_PAGE_SIZE_BYTES - 1)) == 0)
           << std::endl;
        std::cerr << ss.str ();
      }
      #endif

    #endif

  } else { // realloc

    *ptr = realloc (*ptr, new_sz);

    if (WARN_ON_REALLOC != 0)
    {
      std::stringstream ss;
      ss << "Thread[" << omp_get_thread_num() << "] realloc! "
         << "rc = " << ptr << ", PageSize = " << sysconf(_SC_PAGESIZE)
         << ", Scalar Size: " << scalar_size
         << ", requested_count = " << minimum_count << ", NEW count = " << new_count
                                                    << ", NEW sz = " << new_sz << std::endl
         << "Ptr: " << (*ptr)
         << ", Alligned: " << std::boolalpha << (((uintptr_t)(*ptr) & (uintptr_t)(MY_PAGE_SIZE_BYTES - 1)) == 0)
         << std::endl;
      std::cerr << ss.str ();
    }
  }

  return(new_count);
}


template<typename lno_view_t,
         typename lno_nnz_view_t,
         typename scalar_view_t>
static
void parallel_region_copy(const int thread_max, // maybe can use OpenMP
                          const size_t M,       // I think size_t should be replaced with LO, GO size things below to the parallel linear alg level`
                          const size_t nnz_thread_start,
                          lno_view_t     & row_mapC,
                          lno_nnz_view_t & entriesC,
                          scalar_view_t  & valuesC,
                          UnmanagedView<lno_view_t>     & tl_rowptr,
                          UnmanagedView<lno_nnz_view_t> & tl_colind,
                          UnmanagedView<scalar_view_t>  & tl_values) {
  typedef typename lno_view_t::value_type    LO;
  typedef typename scalar_view_t::value_type SC;

  // fuse the copy into the functor

  const double thread_chunk = (double)(M) / thread_max;
  const uint32_t tid = omp_get_thread_num ();
  size_t my_thread_start =  tid * thread_chunk;
  size_t my_thread_stop  = tid == thread_max-1 ? M : (tid+1)*thread_chunk;
  size_t my_thread_m     = my_thread_stop - my_thread_start;

  #if ENABLE_RAW_POINTERS == 1
  __restrict LO * ents_ = entriesC.data();
  __restrict SC * vals_ = valuesC.data();
  __restrict size_t * rows_ = row_mapC.data();

  const __restrict LO     * tl_col_ = tl_colind.data();
  const __restrict SC     * tl_val_ = tl_values.data();
  const __restrict size_t * tl_row_ = tl_rowptr.data();
  #endif

  // Copy out, we still know our thread limits, because we are in the same functor that did the the multiply

  for (size_t i = my_thread_start; i < my_thread_stop; i++) {
    size_t ii = i - my_thread_start;
    // Rowptr
    //row_mapC(i)
    #if ENABLE_RAW_POINTERS == 1
    rows_[i] = nnz_thread_start + tl_row_[ii];
    #else
    row_mapC(i) = nnz_thread_start + tl_rowptr(ii);
    #endif

    if (i==M-1) {
      //row_mapC(m)
      #if ENABLE_RAW_POINTERS == 1
      rows_[M] = nnz_thread_start + tl_row_[ii+1];
      #else
      row_mapC(M) = nnz_thread_start + tl_rowptr(ii+1);
      #endif
    }

    // Colind / Values
     #if ENABLE_RAW_POINTERS == 1
    for(size_t j = tl_row_[ii]; j<tl_row_[ii+1]; j++) {
      ents_[nnz_thread_start + j] = tl_col_[j];
      vals_[nnz_thread_start + j] = tl_val_[j];
    }
    #else
    for(size_t j = tl_rowptr(ii); j<tl_rowptr(ii+1); j++) {
      entriesC(nnz_thread_start + j) = tl_colind(j);
      valuesC(nnz_thread_start + j)  = tl_values(j);
    }
    #endif
  }

  //Free the unamanged views
  if(tl_rowptr.data()) free(tl_rowptr.data());
  if(tl_colind.data()) free(tl_colind.data());
  if(tl_values.data()) free(tl_values.data());
}


template<typename scalar_type>
static
void init_view1D (const scalar_type initial_value,
                  scalar_type * data,
                  const size_t N_) {

   __restrict scalar_type * data_ = data;

  if (N_ <=  std::numeric_limits<uint32_t>::max () ) {
    const uint32_t N = static_cast<uint32_t>(N_);
    #pragma ivdep
    for (uint32_t i=0; i < N; ++i) {
      data_[i] =initial_value;
    }
  } else {
    #pragma ivdep
    for (size_t i=0; i < N_; ++i) {
      data_[i] = initial_value;
    }
  }

} // init_view


} // end anonymous namespace
#endif //defined (HAVE_TPETRA_INST_OPENMP)


namespace Tpetra {

namespace MatrixMatrix{

namespace ExtraKernels{


template<class CrsMatrixType>
size_t C_estimate_nnz_per_row(CrsMatrixType & A, CrsMatrixType &B){
  // Follows the NZ estimate in ML's ml_matmatmult.c
  size_t Aest = 100, Best=100;
  if (A.getNodeNumEntries() > 0)
    Aest = (A.getNodeNumRows() > 0)?  A.getNodeNumEntries()/A.getNodeNumRows() : 100;
  if (B.getNodeNumEntries() > 0)
    Best = (B.getNodeNumRows() > 0) ? B.getNodeNumEntries()/B.getNodeNumRows() : 100;

  size_t nnzperrow = (size_t)(sqrt((double)Aest) + sqrt((double)Best) - 1);
  nnzperrow *= nnzperrow;

  return nnzperrow;
}

#if defined (HAVE_TPETRA_INST_OPENMP)

template<class Scalar,
         class LocalOrdinal,
         class GlobalOrdinal,
         class LocalOrdinalViewType,
         bool copy_out = true>
struct LTGFusedCopyFunctor {
private:
  typedef LTGFusedCopyFunctor<Scalar, LocalOrdinal, GlobalOrdinal, LocalOrdinalViewType, copy_out> this_type;
public:

  // Lots and lots of typedefs
  typedef typename Kokkos::Compat::KokkosOpenMPWrapperNode Node;
  typedef typename Tpetra::CrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>::local_matrix_type KCRS;
  //  typedef typename KCRS::device_type device_t;
  typedef typename KCRS::StaticCrsGraphType graph_t;
  typedef typename graph_t::row_map_type::non_const_type lno_view_t;
  typedef typename graph_t::row_map_type::const_type c_lno_view_t;
  typedef typename graph_t::entries_type::non_const_type lno_nnz_view_t;
  typedef typename KCRS::values_type::non_const_type scalar_view_t;

  // Unmanaged versions of the above
  typedef UnmanagedView<lno_view_t> u_lno_view_t;
  typedef UnmanagedView<lno_nnz_view_t> u_lno_nnz_view_t;
  typedef UnmanagedView<scalar_view_t> u_scalar_view_t;

  typedef UnmanagedView< Kokkos::View<size_t*, typename u_lno_view_t::memory_space> > u_status_array_view_t;
  typedef Scalar            SC;
  typedef LocalOrdinal      LO;
  typedef GlobalOrdinal     GO;
  typedef Node              NO;
  typedef Map<LO,GO,NO>                     map_type;

  // NOTE (mfh 15 Sep 2017) This is specifically only for
  // execution_space = Kokkos::OpenMP, so we neither need nor want
  // KOKKOS_LAMBDA (with its mandatory __device__ marking).
  typedef NO::execution_space execution_space;
  typedef Kokkos::RangePolicy<execution_space, size_t> range_type;


  const bool COPY_OUT = copy_out;

  const LocalOrdinalViewType & targetMapToOrigRow;
  const LocalOrdinalViewType & targetMapToImportRow;
  const LocalOrdinalViewType & Bcol2Ccol;
  const LocalOrdinalViewType & Icol2Ccol;

  // All of the invalid guys
  const LO LO_INVALID;
  const SC SC_ZERO;
  const size_t INVALID;

  // Grab the  Kokkos::SparseCrsMatrices & inner stuff
  //const KCRS & Amat;
  //const KCRS & Bmat;

  const c_lno_view_t  Arowptr;
  const c_lno_view_t  Browptr;
  const lno_nnz_view_t  Acolind;
  const lno_nnz_view_t  Bcolind;
  const scalar_view_t  Avals;
  const scalar_view_t  Bvals;
  size_t b_max_nnz_per_row;

  // Sizes
  Teuchos::RCP<const map_type> Ccolmap;
  const size_t m;
  const size_t n;
  size_t Cest_nnz_per_row;

  // Get my node / thread info (right from openmp or parameter list)
  const size_t thread_max;

  // Thread-local memory
  //Kokkos::View<u_lno_view_t*> tl_rowptr;
  //Kokkos::View<u_lno_nnz_view_t*> tl_colind;
  //Kokkos::View<u_scalar_view_t*> tl_values;

  c_lno_view_t  Irowptr;
  lno_nnz_view_t Icolind;
  scalar_view_t  Ivals;

  // used for final construction
  lno_view_t     row_mapC;
  lno_nnz_view_t entriesC;
  scalar_view_t  valuesC;

  LTGFusedCopyFunctor(CrsMatrixStruct<Scalar, LocalOrdinal, GlobalOrdinal, Kokkos::Compat::KokkosOpenMPWrapperNode>& Aview,
                      CrsMatrixStruct<Scalar, LocalOrdinal, GlobalOrdinal, Kokkos::Compat::KokkosOpenMPWrapperNode>& Bview,
                      const LocalOrdinalViewType & targetMapToOrigRow_,
                      const LocalOrdinalViewType & targetMapToImportRow_,
                      const LocalOrdinalViewType & Bcol2Ccol_,
                      const LocalOrdinalViewType & Icol2Ccol_,
                      CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Kokkos::Compat::KokkosOpenMPWrapperNode>& C,
                      const size_t thread_max_,
                      const Teuchos::RCP<Teuchos::ParameterList>& params) :
                        targetMapToOrigRow (targetMapToOrigRow_),
                        targetMapToImportRow (targetMapToImportRow_),
                        Bcol2Ccol (Bcol2Ccol_),
                        Icol2Ccol (Icol2Ccol_),
                        // All of the invalid guys
                        LO_INVALID (Teuchos::OrdinalTraits<LO>::invalid()),
                        SC_ZERO (Teuchos::ScalarTraits<Scalar>::zero()),
                        INVALID (Teuchos::OrdinalTraits<size_t>::invalid()),
                        // Grab the  Kokkos::SparseCrsMatrices & inner stuff
                        Arowptr (Aview.origMatrix->getLocalMatrix().graph.row_map),
                        Browptr (Bview.origMatrix->getLocalMatrix().graph.row_map),
                        Acolind (Aview.origMatrix->getLocalMatrix().graph.entries),
                        Bcolind (Bview.origMatrix->getLocalMatrix().graph.entries),
                        Avals (Aview.origMatrix->getLocalMatrix().values),
                        Bvals (Bview.origMatrix->getLocalMatrix().values),
                        b_max_nnz_per_row (Bview.origMatrix->getNodeMaxNumRowEntries()),
                        Ccolmap (C.getColMap()),
                        m (Aview.origMatrix->getNodeNumRows()),
                        n (Ccolmap->getNodeNumElements()),
                        Cest_nnz_per_row (2*C_estimate_nnz_per_row(*Aview.origMatrix,*Bview.origMatrix)),
                        // Get my node / thread info (right from openmp or parameter list)
                        thread_max (thread_max_)
                        // Thread-local memory
                        //tl_rowptr (Kokkos::View<u_lno_view_t*> ("top_rowptr", thread_max)),
                        //tl_colind (Kokkos::View<u_lno_nnz_view_t*> ("top_colind", thread_max)),
                        //tl_values (Kokkos::View<u_scalar_view_t*> ("top_values", thread_max)),
                        // used for final construction
                        //row_mapC("non_const_lnow_row", m + 1)

  {
    if(!Bview.importMatrix.is_null()) {
      Irowptr = Bview.importMatrix->getLocalMatrix().graph.row_map;
      Icolind = Bview.importMatrix->getLocalMatrix().graph.entries;
      Ivals   = Bview.importMatrix->getLocalMatrix().values;
      b_max_nnz_per_row = std::max(b_max_nnz_per_row,Bview.importMatrix->getNodeMaxNumRowEntries());
    }

    //std::cerr << "Thread Max: " << thread_max << std::endl;
  }

  ~LTGFusedCopyFunctor() {
  }

  void run () {
  size_t * thread_nnz_array = (size_t*) malloc(sizeof(size_t)* (thread_max+1));
  //::init_view1D (INVALID, thread_nnz_array, (thread_max+1)); 
  #pragma omp parallel
  {
    // Each team will process this chunk of work. Could change this...
    // It would probably be better to use a construct that allows better dynamic work partitioning
    // E.g., a parallel for w/dynamic, but that makes indexing the CSR column entries harder
    // (CSR_ip assumes that blocks are in order)
    // If shared ordered list was used, blocks of rows could be
    // Thread coordination stuff
    const uint32_t tid = omp_get_thread_num ();
    double thread_chunk = (double)(m) / thread_max;
    size_t my_thread_start =  tid * thread_chunk;
    size_t my_thread_stop  = tid == thread_max-1 ? m : (tid+1)*thread_chunk;
    size_t my_thread_m     = my_thread_stop - my_thread_start;

    // Size estimate
    size_t CSR_alloc = (size_t) (my_thread_m*Cest_nnz_per_row);

    // Allocations
    void * ptr = nullptr;
    size_t sz = 0;

    // alloc status array
    sz = ::grow_ptr (n, &ptr, sizeof(typename u_status_array_view_t::value_type));
    u_status_array_view_t c_status( (typename u_status_array_view_t::data_type) ptr, sz);
    // initialize to INVALID
    ::init_view1D (INVALID, c_status.data(), sz);

    // rowptr
    ptr = nullptr;
    sz = ::grow_ptr ((my_thread_m+1), &ptr, sizeof(typename u_lno_view_t::value_type));
    u_lno_view_t Crowptr( (typename u_lno_view_t::data_type) ptr, sz);

    // colind
    ptr = nullptr;
    size_t cind_count = ::grow_ptr (CSR_alloc, &ptr, sizeof(typename u_lno_nnz_view_t::value_type), true);
    u_lno_nnz_view_t Ccolind( (typename u_lno_nnz_view_t::data_type) ptr, cind_count);

    // cvals
    ptr = nullptr;
    size_t cval_count = ::grow_ptr (CSR_alloc, &ptr, sizeof(typename u_scalar_view_t::value_type), true);
    u_scalar_view_t Cvals( (typename u_scalar_view_t::data_type) ptr, cval_count);


    // For each row of A/C
    size_t CSR_ip = 0, OLD_ip = 0;

    #if ENABLE_NESTED == 1
    // start a parallel region... way up here!
    volatile int team_lock = 0;
    #pragma omp parallel
    {
      const int ht_tid = omp_get_thread_num();
      const int parent_tid = omp_get_ancestor_thread_num(1);
      const int team_size = omp_get_team_size(2);
    #endif

      for (size_t i = my_thread_start; i < my_thread_stop; ++i) {
        // mfh 27 Sep 2016: m is the number of rows in the input matrix A
        // on the calling process.
        #if ENABLE_NESTED == 1
        if (ht_tid == 0)
        {
          Crowptr(i-my_thread_start) = CSR_ip;
        }
        #else
        Crowptr(i-my_thread_start) = CSR_ip;
        #endif


        // mfh 27 Sep 2016: For each entry of A in the current row of A

        for (size_t k = Arowptr(i); k < Arowptr(i+1); k++) {
          LO Aik  = Acolind(k); // local column index of current entry of A
          const SC Aval = Avals(k);   // value of current entry of A
          if (Aval == SC_ZERO)
            continue; // skip explicitly stored zero values in A

          if (targetMapToOrigRow(Aik) != LO_INVALID) {
            // mfh 27 Sep 2016: If the entry of targetMapToOrigRow
            // corresponding to the current entry of A is populated, then
            // the corresponding row of B is in B_local (i.e., it lives on
            // the calling process).

            // map the column of A (Aik) to a row in B
            const size_t Bk = Teuchos::as<size_t>(targetMapToOrigRow(Aik));

            // mfh 27 Sep 2016: Go through all entries in that row of B_local.
            #if ENABLE_NESTED == 1
            for (size_t j = Browptr(Bk) + ht_tid; j < Browptr(Bk+1); j += team_size) {
            #else
            for (size_t j = Browptr(Bk);          j < Browptr(Bk+1); ++j) {
            #endif
              const LO Bkj = Bcolind(j);
              const LO Cij = Bcol2Ccol(Bkj);

              if (c_status[Cij] == INVALID || c_status[Cij] < OLD_ip) {
                // New entry
                #if ENABLE_NESTED == 1
                  c_status[Cij] = Kokkos::atomic_fetch_add( &CSR_ip, 1);
                #else
                  c_status[Cij]   = CSR_ip;
                  CSR_ip++;
                #endif
                Ccolind(c_status[Cij]) = Cij;

                Cvals(c_status[Cij]) = Aval*Bvals(j);
              } else {
                Cvals(c_status[Cij]) += Aval*Bvals(j);
              }
            }

          } else {
            if (WARN_ON_REALLOC != 0) {
              std::stringstream ss;
              ss << "Thread[" << omp_get_thread_num() << "] Hit the branch!" << std::endl;
              std::cerr << ss.str ();
            }

            // mfh 27 Sep 2016: If the entry of targetMapToOrigRow
            // corresponding to the current entry of A NOT populated (has
            // a flag "invalid" value), then the corresponding row of B is
            // in B_local (i.e., it lives on the calling process).

            // Remote matrix
            size_t Ik = Teuchos::as<size_t>(targetMapToImportRow(Aik));
            for (size_t j = Irowptr(Ik); j < Irowptr(Ik+1); ++j) {
              LO Ikj = Icolind(j);
              LO Cij = Icol2Ccol(Ikj);

              if (c_status[Cij] == INVALID || c_status[Cij] < OLD_ip){
                // New entry
                c_status[Cij]   = CSR_ip;
                Ccolind(CSR_ip) = Cij;
                Cvals(CSR_ip)   = Aval*Ivals(j);
                CSR_ip++;

              } else {
                Cvals(c_status[Cij]) += Aval*Ivals(j);
              }
            }
          }
        } // if Aik maps to a local or remote map

        #if ENABLE_NESTED == 1
        // need to sync the HTs there
        {
          #if ENABLE_USE_OMP_BARRIER == 1
            #pragma omp barrier
            #pragma omp single
            // Resize for next pass if needed
            {
              // estimate the the number of entires we will need
              const size_t estimated_entries = CSR_ip + std::min(n,(Arowptr(i+2)-Arowptr(i+1))*b_max_nnz_per_row);
              if (i+1 < my_thread_stop && estimated_entries > cval_count) {
                void * ptr = Cvals.data();
                cval_count = ::grow_ptr (estimated_entries, &ptr, sizeof(typename decltype(Cvals)::value_type), true);
                Cvals = UmanagedView((decltype(Cvals)::data_type) ptr, cval_count);
              }
              if (i+1 < my_thread_stop && estimated_entries > cind_count) {
                void * ptr = Ccolind.data();
                cind_count = ::grow_ptr (estimated_entries, &ptr, sizeof(typename decltype(Ccolind)::value_type), true);
                Ccolind = UmanagedView((decltype(Ccolind)::data_type) ptr, cind_count);
              }
              OLD_ip = CSR_ip;
            } // implicit barrier here

          #else
            const int32_t lock_value = ::fetch_and_add( &team_lock, int(1));  //Kokkos::atomic_fetch_add( &team_lock, 1 );
          // if team_lock == team_size, then this thread could do the realloc stuff
          if ( lock_value == (team_size-1)) {
            std::stringstream ss;
            ss << "Thread[" << parent_tid << "," << omp_get_thread_num() << "] Got a lock!" << lock_value
                                                                         << ", team_lock = " << team_lock
                                                                         << ", team_size = " << team_size << std::endl;
            std::cerr << ss.str ();
            // Resize for next pass if needed
            {
              // estimate the the number of entires we will need
              const size_t estimated_entries = CSR_ip + std::min(n,(Arowptr(i+2)-Arowptr(i+1))*b_max_nnz_per_row);
              if (i+1 < my_thread_stop && estimated_entries > cval_count) {
                void * ptr = Cvals.data();
                cval_count = ::grow_ptr (estimated_entries, &ptr, sizeof(typename decltype(Cvals)::value_type), true);
                Cvals = UmanagedView((decltype(Cvals)::data_type) ptr, cval_count);
              }
              if (i+1 < my_thread_stop && estimated_entries > cind_count) {
                void * ptr = Ccolind.data();
                cind_count = ::grow_ptr (estimated_entries, &ptr, sizeof(typename decltype(Ccolind)::value_type), true);
                Ccolind = UmanagedView((decltype(Ccolind)::data_type) ptr, cind_count);
              }
              OLD_ip = CSR_ip;
            }

            // need a fence
            ss.str("");
                ss << "Thread[" << parent_tid << "," << omp_get_thread_num() << "] Release lock!" << lock_value
                                                                             << ", team_lock = " << team_lock
                                                                             << ", team_size = " << team_size << std::endl;
                std::cerr << ss.str ();

            Kokkos::atomic_compare_exchange(&team_lock, team_lock, 0);
          } else {
            std::stringstream ss;
            ss << "Thread[" << parent_tid << ","  << omp_get_thread_num() << "] Blocking! lock_value = " << lock_value
                                                                             << ", team_lock = " << team_lock
                                                                             << ", team_size = " << team_size << std::endl;;
            std::cerr << ss.str();
            volatile int32_t * t_l_ = &team_lock;

            while ( *(t_l_) != 0) {
              t_l_ = &team_lock;
            }// yuck, can we yield?
            ss.str("");
            ss << "Thread[" << parent_tid << "," << omp_get_thread_num() << "] unblocking!" << lock_value
                                                                             << ", team_lock = " << team_lock
                                                                             << ", team_size = " << team_size << std::endl;
            std::cerr << ss.str();
          }
          #endif
        }
        #else
        // Resize for next pass if needed
        {
          // estimate the the number of entires we will need
          const size_t estimated_entries = CSR_ip + std::min(n,(Arowptr(i+2)-Arowptr(i+1))*b_max_nnz_per_row);
          if (i+1 < my_thread_stop && estimated_entries > cval_count) {
            typedef decltype(Cvals) v_t;
            typedef typename v_t::value_type my_type;
            typedef typename v_t::data_type my_data_type;

            void * ptr = Cvals.data();
            cval_count = ::grow_ptr (estimated_entries, &ptr, sizeof(my_type), true);
            Cvals = v_t( (my_data_type) ptr, cval_count);
          }
          if (i+1 < my_thread_stop && estimated_entries > cind_count) {
            typedef decltype(Ccolind) v_t;
            typedef typename v_t::value_type my_type;
            typedef typename v_t::data_type my_data_type;

            void * ptr = Ccolind.data();
            cind_count = ::grow_ptr (estimated_entries, &ptr, sizeof(my_type), true);
            Ccolind = v_t( (my_data_type) ptr, cind_count);
          }
          OLD_ip = CSR_ip;
        }
        #endif  // ENABLE_NESTED == 1 block (resizes/syncs threads)
     }
    #if ENABLE_NESTED  == 1
    // close parallel region
    }
    #endif

    if (c_status.data()) free(c_status.data());
    Crowptr(my_thread_m) = CSR_ip;

    if (COPY_OUT) {
      // share our nnz in the global array
      thread_nnz_array[tid] = CSR_ip;

      // you must synchronize before looping here
      #pragma omp barrier

      #pragma omp single
      {
        // Generate the starting nnz number per thread
        size_t c_nnz_size = 0;
        size_t sum = 0;
        for (size_t i=0; i < thread_max; ++i){
          // remember what this thread contributes
          size_t threads_nnz = thread_nnz_array[i];
          // update this thread with their starting point
          thread_nnz_array[i] = sum;
          // add the threads contribution
          sum += threads_nnz;
        }
        thread_nnz_array[thread_max] = sum;
        c_nnz_size = thread_nnz_array[thread_max];

        // Allocate output
        lno_nnz_view_t entriesC_(Kokkos::ViewAllocateWithoutInitializing("entriesC"), c_nnz_size); entriesC = entriesC_;
        scalar_view_t  valuesC_(Kokkos::ViewAllocateWithoutInitializing("valuesC"), c_nnz_size);  valuesC = valuesC_;
        lno_view_t row_mapC_(Kokkos::ViewAllocateWithoutInitializing("non_const_lnow_row"), m + 1); row_mapC = row_mapC_;
      } // implicit barrier

      // call the copy, this deallocates the unmanaged views
      parallel_region_copy(thread_max, // maybe can use OpenMP
                           m,       // I think size_t should be replaced with LO, GO size things below to the parallel linear alg level`
                           thread_nnz_array[tid],
                           row_mapC,
                           entriesC,
                           valuesC,
                           Crowptr,
                           Ccolind,
                           Cvals);
   } // fused copy out if
  } // omp parallel
  if (thread_nnz_array) free(thread_nnz_array);
 } // run() function body
}; // end of Functor class


/*
template<class Scalar,
         class LocalOrdinal,
         class GlobalOrdinal,
         class LocalOrdinalViewType,
         bool copy_out = true>
struct LTGFusedCopyFunctor {
private:
  typedef LTGFusedCopyFunctor<Scalar, LocalOrdinal, GlobalOrdinal, LocalOrdinalViewType, copy_out> this_type;
public:

  // Lots and lots of typedefs
  typedef typename Kokkos::Compat::KokkosOpenMPWrapperNode Node;
  typedef typename Tpetra::CrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>::local_matrix_type KCRS;
  //  typedef typename KCRS::device_type device_t;
  typedef typename KCRS::StaticCrsGraphType graph_t;
  typedef typename graph_t::row_map_type::non_const_type lno_view_t;
  typedef typename graph_t::row_map_type::const_type c_lno_view_t;
  typedef typename graph_t::entries_type::non_const_type lno_nnz_view_t;
  typedef typename KCRS::values_type::non_const_type scalar_view_t;

  // Unmanaged versions of the above
  typedef UnmanagedView<lno_view_t> u_lno_view_t;
  typedef UnmanagedView<lno_nnz_view_t> u_lno_nnz_view_t;
  typedef UnmanagedView<scalar_view_t> u_scalar_view_t;

  typedef Scalar            SC;
  typedef LocalOrdinal      LO;
  typedef GlobalOrdinal     GO;
  typedef Node              NO;
  typedef Map<LO,GO,NO>                     map_type;

  // NOTE (mfh 15 Sep 2017) This is specifically only for
  // execution_space = Kokkos::OpenMP, so we neither need nor want
  // KOKKOS_LAMBDA (with its mandatory __device__ marking).
  typedef NO::execution_space execution_space;
  typedef Kokkos::RangePolicy<execution_space, size_t> range_type;


  const bool COPY_OUT = copy_out;

  const LocalOrdinalViewType & targetMapToOrigRow;
  const LocalOrdinalViewType & targetMapToImportRow;
  const LocalOrdinalViewType & Bcol2Ccol;
  const LocalOrdinalViewType & Icol2Ccol;

  // All of the invalid guys
  const LO LO_INVALID;
  const SC SC_ZERO;
  const size_t INVALID;

  // Grab the  Kokkos::SparseCrsMatrices & inner stuff
  const KCRS & Amat;
  const KCRS & Bmat;

  const c_lno_view_t Arowptr;
  const c_lno_view_t Browptr;
  const lno_nnz_view_t Acolind;
  const lno_nnz_view_t Bcolind;
  const scalar_view_t Avals;
  const scalar_view_t Bvals;
  size_t b_max_nnz_per_row;

  // Sizes
  Teuchos::RCP<const map_type> Ccolmap;
  const size_t m;
  const size_t n;
  size_t Cest_nnz_per_row;

  // Get my node / thread info (right from openmp or parameter list)
  const size_t thread_max;
  double thread_chunk;

  // Thread-local memory
  Kokkos::View<u_lno_view_t*> tl_rowptr;
  Kokkos::View<u_lno_nnz_view_t*> tl_colind;
  Kokkos::View<u_scalar_view_t*> tl_values;

  c_lno_view_t  Irowptr;
  lno_nnz_view_t Icolind;
  scalar_view_t  Ivals;

  // used for final construction
  lno_view_t     row_mapC;
  lno_nnz_view_t entriesC;
  scalar_view_t  valuesC;

  LTGFusedCopyFunctor(CrsMatrixStruct<Scalar, LocalOrdinal, GlobalOrdinal, Kokkos::Compat::KokkosOpenMPWrapperNode>& Aview,
                      CrsMatrixStruct<Scalar, LocalOrdinal, GlobalOrdinal, Kokkos::Compat::KokkosOpenMPWrapperNode>& Bview,
                      const LocalOrdinalViewType & targetMapToOrigRow_,
                      const LocalOrdinalViewType & targetMapToImportRow_,
                      const LocalOrdinalViewType & Bcol2Ccol_,
                      const LocalOrdinalViewType & Icol2Ccol_,
                      CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Kokkos::Compat::KokkosOpenMPWrapperNode>& C,
                      const size_t thread_max_,
                      const Teuchos::RCP<Teuchos::ParameterList>& params) :
                        targetMapToOrigRow (targetMapToOrigRow_),
                        targetMapToImportRow (targetMapToImportRow_),
                        Bcol2Ccol (Bcol2Ccol_),
                        Icol2Ccol (Icol2Ccol_),
                        // All of the invalid guys
                        LO_INVALID (Teuchos::OrdinalTraits<LO>::invalid()),
                        SC_ZERO (Teuchos::ScalarTraits<Scalar>::zero()),
                        INVALID (Teuchos::OrdinalTraits<size_t>::invalid()),
                        // Grab the  Kokkos::SparseCrsMatrices & inner stuff
                        Amat (Aview.origMatrix->getLocalMatrix()),
                        Bmat (Bview.origMatrix->getLocalMatrix()),
                        Arowptr (Amat.graph.row_map),
                        Browptr (Bmat.graph.row_map),
                        Acolind (Amat.graph.entries),
                        Bcolind (Bmat.graph.entries),
                        Avals (Amat.values),
                        Bvals (Bmat.values),
                        b_max_nnz_per_row (Bview.origMatrix->getNodeMaxNumRowEntries()),
                        Ccolmap (C.getColMap()),
                        m (Aview.origMatrix->getNodeNumRows()),
                        n (Ccolmap->getNodeNumElements()),
                        Cest_nnz_per_row (2*C_estimate_nnz_per_row(*Aview.origMatrix,*Bview.origMatrix)),
                        // Get my node / thread info (right from openmp or parameter list)
                        thread_max (thread_max_),
                        // Thread-local memory
                        tl_rowptr (Kokkos::View<u_lno_view_t*> ("top_rowptr", thread_max)),
                        tl_colind (Kokkos::View<u_lno_nnz_view_t*> ("top_colind", thread_max)),
                        tl_values (Kokkos::View<u_scalar_view_t*> ("top_values", thread_max)),
                        // used for final construction
                        row_mapC("non_const_lnow_row", m + 1)

  {
    if(!Bview.importMatrix.is_null()) {
      Irowptr = Bview.importMatrix->getLocalMatrix().graph.row_map;
      Icolind = Bview.importMatrix->getLocalMatrix().graph.entries;
      Ivals   = Bview.importMatrix->getLocalMatrix().values;
      b_max_nnz_per_row = std::max(b_max_nnz_per_row,Bview.importMatrix->getNodeMaxNumRowEntries());
    }

    thread_chunk = (double)(m) / thread_max;
  }

  ~LTGFusedCopyFunctor() {
    //Free the unamanged views
    for(size_t i=0; i<thread_max; i++) {
      if(tl_rowptr(i).data()) free(tl_rowptr(i).data());
      if(tl_colind(i).data()) free(tl_colind(i).data());
      if(tl_values(i).data()) free(tl_values(i).data());
    }
  }

  void run () {
  #pragma omp parallel
  {
    // Each team will process this chunk of work. Could change this...
    // It would probably be better to use a construct that allows better dynamic work partitioning
    // E.g., a parallel for w/dynamic, but that makes indexing the CSR column entries harder
    // (CSR_ip assumes that blocks are in order)
    // If shared ordered list was used, blocks of rows could be
    // Thread coordination stuff
    const uint32_t tid = omp_get_thread_num ();
    size_t my_thread_start =  tid * thread_chunk;
    size_t my_thread_stop  = tid == thread_max-1 ? m : (tid+1)*thread_chunk;
    size_t my_thread_m     = my_thread_stop - my_thread_start;

    // Size estimate
    size_t CSR_alloc = (size_t) (my_thread_m*Cest_nnz_per_row);

    // Allocations
    //std::map<LO,size_t>>
    std::vector<size_t> c_status(n,INVALID);


    #if HUGE_PAGE_MALLOC == 1
    void * cc_ptr;
    int rc;

    // CSR alloc isn't accurate anymore.. need to choose the smallest of these
    // This is terse, but it makes it clear we are padding to some page size.
    size_t colind_csr_sz = MY_PAGE_SIZE_BYTES*((CSR_alloc*sizeof(LO)/MY_PAGE_SIZE_BYTES) + ((CSR_alloc*sizeof(LO)) % MY_PAGE_SIZE_BYTES == 0 ? 0 : 1));
    size_t colval_csr_sz = MY_PAGE_SIZE_BYTES*((CSR_alloc*sizeof(SC)/MY_PAGE_SIZE_BYTES) + ((CSR_alloc*sizeof(SC)) % MY_PAGE_SIZE_BYTES == 0 ? 0 : 1));

    size_t colind_csr_count = colind_csr_sz / sizeof(LO);
    size_t colval_csr_count = colval_csr_sz / sizeof(SC);

    rc = posix_memalign(&cc_ptr, MY_PAGE_SIZE_BYTES, (my_thread_m+1)*sizeof(LO)); // rc == 0 or DIE
    u_lno_view_t Crowptr((typename u_lno_view_t::data_type) cc_ptr, my_thread_m+1);


    // pass the true size (page multiples), we may not need to realloc the col indices if they are 32bit
    rc = posix_memalign(&cc_ptr, MY_PAGE_SIZE_BYTES, colind_csr_sz); // rc == 0 or DIE
    u_lno_nnz_view_t Ccolind((typename u_lno_nnz_view_t::data_type) cc_ptr, colind_csr_count);
    {
      std::stringstream ss;
      ss << "rc: "<< rc << ", ptr: " << cc_ptr << ", PG: " << MY_PAGE_SIZE_BYTES << ", sz: " << colind_csr_sz << ", count: " << colind_csr_count
         << ", div: " << (colind_csr_sz/colind_csr_count) << std::endl;
      std::cerr << ss.str();
    }
    rc = posix_memalign(&cc_ptr, MY_PAGE_SIZE_BYTES, colval_csr_sz); // rc == 0 or DIE
    u_scalar_view_t Cvals((typename u_scalar_view_t::data_type) cc_ptr, colval_csr_count);
    {
      std::stringstream ss;
      ss << "rc: "<< rc << ", ptr: " << cc_ptr << ", PG: " << MY_PAGE_SIZE_BYTES << ", sz: " << colval_csr_sz << ", count: " << colval_csr_count
         << ", div: " << (colval_csr_sz/colval_csr_count) << std::endl;
      std::cerr << ss.str();
    }
    #else

    void * ptr;
    //posix_memalign(&ptr, MY_PAGE_SIZE_BYTES, (my_thread_m+1)*sizeof(LO));
    u_lno_view_t Crowptr((typename u_lno_view_t::data_type) malloc(sizeof(LO)*(my_thread_m+1)),my_thread_m+1);

    //posix_memalign(&ptr, MY_PAGE_SIZE_BYTES, CSR_alloc*sizeof(LO));
    u_lno_nnz_view_t Ccolind((typename u_lno_nnz_view_t::data_type) malloc(sizeof(LO)*CSR_alloc), CSR_alloc);
    //posix_memalign(&ptr, MY_PAGE_SIZE_BYTES, CSR_alloc*sizeof(SC));
    u_scalar_view_t Cvals((typename u_scalar_view_t::data_type) malloc(sizeof(SC)*CSR_alloc), CSR_alloc);
    std::cerr << "csr_alloc: " << CSR_alloc << std::endl;
    #endif

    #if HUGE_PAGE_MALLOC == 1 && REALLOC_USE_PAGES != 1
    CSR_alloc = std::min(colind_csr_count, colval_csr_count);
    #endif

    // For each row of A/C
    size_t CSR_ip = 0, OLD_ip = 0;

    #if ENABLE_BLOCKED_COLS == 1
    constexpr uint32_t VAL_BLOCK_SIZE = 16;
    typedef struct team_workspace {
      SC vals[VAL_BLOCK_SIZE];
      LO idxs[VAL_BLOCK_SIZE];
    } team_workspace_t;
    #endif

    #if ENABLE_NESTED == 1
    // start a parallel region... way up here!
    volatile int team_lock = 0;
    #pragma omp parallel
    {
      const int ht_tid = omp_get_thread_num();
      const int parent_tid = omp_get_ancestor_thread_num(1);
      const int team_size = omp_get_team_size(2);
    #endif

      #if ENABLE_BLOCKED_COLS == 1
      team_workspace_t blk;
      #endif
//    // nested parallel loop, assumes OMP_NESTED=true
//    // this is intended to allow hardware threads to parallelize across rows,
//    // but stay relatively close in terms of data from A, data from B will still
//    // bounce around..

      for (size_t i = my_thread_start; i < my_thread_stop; ++i) {
        // mfh 27 Sep 2016: m is the number of rows in the input matrix A
        // on the calling process.
        #if ENABLE_NESTED == 1
        if (ht_tid == 0)
        {
          Crowptr(i-my_thread_start) = CSR_ip;
        }
        #else
        Crowptr(i-my_thread_start) = CSR_ip;
        #endif


        // mfh 27 Sep 2016: For each entry of A in the current row of A

        for (size_t k = Arowptr(i); k < Arowptr(i+1); k++) {
          LO Aik  = Acolind(k); // local column index of current entry of A
          const SC Aval = Avals(k);   // value of current entry of A
          if (Aval == SC_ZERO)
            continue; // skip explicitly stored zero values in A

          if (targetMapToOrigRow(Aik) != LO_INVALID) {
            // mfh 27 Sep 2016: If the entry of targetMapToOrigRow
            // corresponding to the current entry of A is populated, then
            // the corresponding row of B is in B_local (i.e., it lives on
            // the calling process).

            // Local matrix
            const size_t Bk = Teuchos::as<size_t>(targetMapToOrigRow(Aik));

            #if ENABLE_BLOCKED_COLS == 1
            uint32_t blk_idx = 0;
            #endif

            // mfh 27 Sep 2016: Go through all entries in that row of B_local.
            for (
                 #if ENABLE_NESTED == 1
                 size_t j = Browptr(Bk) + ht_tid;
                 #else
                 size_t j = Browptr(Bk);
                 #endif
                 j < Browptr(Bk+1);
                 #if ENABLE_BLOCKED_COLS == 1
                   #if ENABLE_NESTED == 1
                     j += team_size, ++blk_idx) {
                   #else
                     ++j, ++blk_idx) {
                   #endif
                 #else
                   #if ENABLE_NESTED == 1
                     j += team_size) {
                   #else
                     ++j) {
                   #endif
                 #endif
              const LO Bkj = Bcolind(j);
              const LO Cij = Bcol2Ccol(Bkj);

              #if ENABLE_BLOCKED_COLS == 1
              if (c_status[Cij] == INVALID || c_status[Cij] < OLD_ip) {
                // New entry
                #if ENABLE_NESTED == 1
                  c_status[Cij] = Kokkos::atomic_fetch_add( &CSR_ip, 1);
                #else
                  c_status[Cij]   = CSR_ip++;
                #endif
                Ccolind(c_status[Cij]) = Cij;
              }

              // check if we exhaust the block size
              if (blk_idx >= VAL_BLOCK_SIZE) {
                for (uint32_t b = 0; b < VAL_BLOCK_SIZE; ++b) {
                  blk.vals[b] = Aik * blk.vals[b];
                }
                for (uint32_t b = 0; b < VAL_BLOCK_SIZE; ++b) {
                  #if ENABLE_NESTED == 1 || ENABLE_FORCE_ATOMIC_ADD == 1
                   Kokkos::atomic_fetch_add( &Cvals(c_status[(blk.idxs[b])]), blk.vals[b]);
                  #else
                   Cvals(c_status[(blk.idxs[b])]) += blk.vals[b];
                  #endif
                }
                blk_idx=0;
              } else {
                blk.vals[blk_idx] = Bvals(j);
                blk.idxs[blk_idx] = Cij;
              }
              #else
              if (c_status[Cij] == INVALID || c_status[Cij] < OLD_ip) {
                // New entry
                #if ENABLE_NESTED == 1
                  c_status[Cij] = Kokkos::atomic_fetch_add( &CSR_ip, 1);
                #else
                  c_status[Cij]   = CSR_ip++;
                #endif
                Ccolind(c_status[Cij]) = Cij;

                Cvals(c_status[Cij]) = Aval*Bvals(j);
              } else {
                Cvals(c_status[Cij]) += Aval*Bvals(j);
              }
              #endif
            }

            // process the blocks
            #if ENABLE_BLOCKED_COLS == 1
            for (uint32_t b = 0; b < blk_idx; ++b) {
              blk.vals[b] = Aik * blk.idxs[b];
            }

            for (uint32_t b = 0; b < blk_idx; ++b) {
              #if ENABLE_FORCE_ATOMIC_ADD == 1
               Kokkos::atomic_fetch_add( &Cvals(c_status[(blk.idxs[b])]), blk.vals[b]);
              #else
               Cvals(c_status[(blk.idxs[b])]) += blk.vals[b];
              #endif
            }
            #endif

          } else {
            if (WARN_ON_REALLOC != 0) {
              std::stringstream ss;
              ss << "Thread[" << omp_get_thread_num() << "] Hit the branch!" << std::endl;
              std::cerr << ss.str ();
            }

            // mfh 27 Sep 2016: If the entry of targetMapToOrigRow
            // corresponding to the current entry of A NOT populated (has
            // a flag "invalid" value), then the corresponding row of B is
            // in B_local (i.e., it lives on the calling process).

            // Remote matrix
            size_t Ik = Teuchos::as<size_t>(targetMapToImportRow(Aik));
            for (size_t j = Irowptr(Ik); j < Irowptr(Ik+1); ++j) {
              LO Ikj = Icolind(j);
              LO Cij = Icol2Ccol(Ikj);

              if (c_status[Cij] == INVALID || c_status[Cij] < OLD_ip){
                // New entry
                c_status[Cij]   = CSR_ip;
                Ccolind(CSR_ip) = Cij;
                Cvals(CSR_ip)   = Aval*Ivals(j);
                CSR_ip++;

              } else {
                Cvals(c_status[Cij]) += Aval*Ivals(j);
              }
            }
          }
        } // if Aik maps to a local or remote map

        #if ENABLE_NESTED == 1
        // need to sync the HTs there
        {
          #if ENABLE_USE_OMP_BARRIER == 1
            #pragma omp barrier
            #pragma omp single
            {
            // Resize for next pass if needed
            if (i+1 < my_thread_stop && CSR_ip + std::min(n,(Arowptr(i+2)-Arowptr(i+1))*b_max_nnz_per_row) > CSR_alloc) {

              if (WARN_ON_REALLOC != 0) {
                std::stringstream ss1;
                ss1 << "Thread[" << omp_get_thread_num() << "] Realloc! CSR_alloc = " << CSR_alloc << ", NEW = " << CSR_alloc*2 << std::endl;
                std::cerr << ss1.str ();
              }
              CSR_alloc *= 2;
              Ccolind = u_lno_nnz_view_t((typename u_lno_nnz_view_t::data_type)realloc(Ccolind.data(),u_lno_nnz_view_t::shmem_size(CSR_alloc)),CSR_alloc);
              Cvals = u_scalar_view_t((typename u_scalar_view_t::data_type)realloc(Cvals.data(),u_scalar_view_t::shmem_size(CSR_alloc)),CSR_alloc);
            } // if
            OLD_ip = CSR_ip;
            }
          #else
            const int32_t lock_value = fetch_and_add( &team_lock, int(1));  //Kokkos::atomic_fetch_add( &team_lock, 1 );
          // if team_lock == team_size, then this thread could do the realloc stuff
          if ( lock_value == (team_size-1)) {
                std::stringstream ss;
                ss << "Thread[" << parent_tid << "," << omp_get_thread_num() << "] Got a lock!" << lock_value
                                                                             << ", team_lock = " << team_lock
                                                                             << ", team_size = " << team_size << std::endl;
                std::cerr << ss.str ();
            // Resize for next pass if needed
            if (i+1 < my_thread_stop && CSR_ip + std::min(n,(Arowptr(i+2)-Arowptr(i+1))*b_max_nnz_per_row) > CSR_alloc) {

              if (WARN_ON_REALLOC != 0) {
                std::stringstream ss1;
                ss1 << "Thread[" << omp_get_thread_num() << "] Realloc! CSR_alloc = " << CSR_alloc << ", NEW = " << CSR_alloc*2 << std::endl;
                std::cerr << ss1.str ();
              }
              CSR_alloc *= 2;
              Ccolind = u_lno_nnz_view_t((typename u_lno_nnz_view_t::data_type)realloc(Ccolind.data(),u_lno_nnz_view_t::shmem_size(CSR_alloc)),CSR_alloc);
              Cvals = u_scalar_view_t((typename u_scalar_view_t::data_type)realloc(Cvals.data(),u_scalar_view_t::shmem_size(CSR_alloc)),CSR_alloc);
            }
            // this is a critical section.. move to top?
            OLD_ip = CSR_ip;
            // need a fence
            ss.str("");
                ss << "Thread[" << parent_tid << "," << omp_get_thread_num() << "] Release lock!" << lock_value
                                                                             << ", team_lock = " << team_lock
                                                                             << ", team_size = " << team_size << std::endl;
                std::cerr << ss.str ();

            Kokkos::atomic_compare_exchange(&team_lock, team_lock, 0);
          } else {
            std::stringstream ss;
            ss << "Thread[" << parent_tid << ","  << omp_get_thread_num() << "] Blocking! lock_value = " << lock_value
                                                                             << ", team_lock = " << team_lock
                                                                             << ", team_size = " << team_size << std::endl;;
            std::cerr << ss.str();
            volatile int32_t * t_l_ = &team_lock;

            while ( *(t_l_) != 0) {
              t_l_ = &team_lock;
            }// yuck, can we yield?
            ss.str("");
            ss << "Thread[" << parent_tid << "," << omp_get_thread_num() << "] unblocking!" << lock_value
                                                                             << ", team_lock = " << team_lock
                                                                             << ", team_size = " << team_size << std::endl;
            std::cerr << ss.str();
          }
          #endif
        }
        #else
        // Resize for next pass if needed

        #if REALLOC_USE_PAGES == 1 && HUGE_PAGE_MALLOC == 1
        // Split this into Colind and Colval, if LO = 32bit, we get more per page
        //  colind_csr_sz = current size in bytes of Ccolind
        //  colind_csr_count = count of addressable items (e.g., LOs or SCs)
        //
        //  colval_csr_sz = (see above)
        //  colval_csr_count (see above)
        const size_t est_ = CSR_ip + std::min(n,(Arowptr(i+2)-Arowptr(i+1))*b_max_nnz_per_row);

        if (i+1 < my_thread_stop && est_ > colind_csr_count) {

          const size_t new_sz = colind_csr_sz + MY_PAGE_SIZE_BYTES;
          const size_t new_count = colind_csr_count + (MY_PAGE_SIZE_BYTES / sizeof(LO));

          if (WARN_ON_REALLOC != 0) {
            std::stringstream ss;
            ss << "Thread[" << omp_get_thread_num() << "] Realloc! colind_csr_count = " << colind_csr_count << ", NEW count = " << new_count
                                                    << ", colind_csr_sz = " << colind_csr_sz << ", NEW sz = " << new_sz << std::endl;
            std::cerr << ss.str ();
          }

          Ccolind = u_lno_nnz_view_t( (typename u_lno_nnz_view_t::data_type) realloc (Ccolind.data(), new_sz), new_count);

          colind_csr_sz = new_sz;
          colind_csr_count = new_count;
        }

        if (i+1 < my_thread_stop && est_ > colval_csr_count) {

          const size_t new_sz = colval_csr_sz + MY_PAGE_SIZE_BYTES;
          const size_t new_count = colval_csr_count + (MY_PAGE_SIZE_BYTES / sizeof(SC));
          if (WARN_ON_REALLOC != 0) {
            std::stringstream ss;
            ss << "Thread[" << omp_get_thread_num() << "] Realloc! colval_csr_count = " << colval_csr_count << ", NEW count = " << new_count
                                                    << ", colval_csr_sz = " << colval_csr_sz << ", NEW sz = " << new_sz << std::endl;
            std::cerr << ss.str ();
          }

          Cvals = u_scalar_view_t((typename u_scalar_view_t::data_type) realloc (Cvals.data(), new_sz), new_count);

          colval_csr_sz = new_sz;
          colval_csr_count = new_sz;
        }
        #else
        if (i+1 < my_thread_stop && CSR_ip + std::min(n,(Arowptr(i+2)-Arowptr(i+1))*b_max_nnz_per_row) > CSR_alloc) {

          if (WARN_ON_REALLOC != 0) {
            std::stringstream ss;
            ss << "Thread[" << omp_get_thread_num() << "] Realloc! CSR_alloc = " << CSR_alloc << ", NEW = " << CSR_alloc*2 << std::endl;
            ss << "u_lno_nnz_view_t::shmem_size(CSR_alloc*2) = " << u_lno_nnz_view_t::shmem_size(CSR_alloc*2) << std::endl
               << "sizeof(LO)*CSR_alloc*2 = " << sizeof(LO)*CSR_alloc*2 << std::endl
               << "u_scalar_view_t::shmem_size(CSR_alloc*2)  = " << u_scalar_view_t::shmem_size(CSR_alloc*2)  << std::endl
               << "sizeof(SC*CSR_alloc*2 = " << sizeof(SC)*CSR_alloc*2 << std::endl;

            std::cerr << ss.str ();
          }
          CSR_alloc *= 2;

          // something weird is going on here.

          Ccolind = u_lno_nnz_view_t((typename u_lno_nnz_view_t::data_type)realloc(Ccolind.data(),u_lno_nnz_view_t::shmem_size(CSR_alloc)),CSR_alloc);
          Cvals = u_scalar_view_t((typename u_scalar_view_t::data_type)realloc(Cvals.data(),u_scalar_view_t::shmem_size(CSR_alloc)),CSR_alloc);
        }
        #endif
        OLD_ip = CSR_ip;
        #endif
      }
    #if ENABLE_NESTED  == 1
    // close parallel region
    }
    #endif

    tl_rowptr(tid) = Crowptr;
    tl_colind(tid) = Ccolind;
    tl_values(tid) = Cvals;
    Crowptr(my_thread_m) = CSR_ip;

    if (copy_out) {
    // fuse the copy into the functor
    #pragma omp barrier
    {
      //
      //  tl_rowptr => Inrowptr,
      //  tl_colind => Incolind,
      //  tl_values => Invalues,
      //

      // Generate the starting nnz number per thread
      // assuming low thread counts, simply replicate this data/computation. It's cheaper than synchronizing
      size_t c_nnz_size = 0;

      // this is throwing an exception
      // Constructing View and initializing data with uninitialized execution space
      // lno_view_t thread_start_nnz("thread_nnz", thread_max+1);
      LO * thread_start_nnz = new LO[thread_max+1];
      //Kokkos::View<LO*, Kokkos::HostSpace> thread_start_nnz("thread_nnz", thread_max+1);

      size_t sum = 0;
      for (size_t i=0; i < thread_max; ++i){
        thread_start_nnz[i] = sum;
        sum += tl_rowptr(i)(tl_rowptr(i).dimension(0)-1);
      }
      thread_start_nnz[thread_max] = sum;
      c_nnz_size = thread_start_nnz[thread_max];

      #pragma omp single
      {
        // Allocate output
        lno_nnz_view_t entriesC_(Kokkos::ViewAllocateWithoutInitializing("entriesC"), c_nnz_size); entriesC = entriesC_;
        scalar_view_t  valuesC_(Kokkos::ViewAllocateWithoutInitializing("valuesC"), c_nnz_size);  valuesC = valuesC_;
      }

      #pragma omp barrier
      {
         // Copy out, we still know our thread limits, because we are in the same functor that did the the multiply
         size_t nnz_thread_start = thread_start_nnz[tid];

         for (size_t i = my_thread_start; i < my_thread_stop; i++) {
           size_t ii = i - my_thread_start;
           // Rowptr
           row_mapC(i) = nnz_thread_start + tl_rowptr(tid)(ii);
           if (i==m-1) {
             row_mapC(m) = nnz_thread_start + tl_rowptr(tid)(ii+1);
           }

           // Colind / Values
           for(size_t j = tl_rowptr(tid)(ii); j<tl_rowptr(tid)(ii+1); j++) {
             entriesC(nnz_thread_start + j) = tl_colind(tid)(j);
             valuesC(nnz_thread_start + j)  = tl_values(tid)(j);
           }
         }
      }
      delete [] thread_start_nnz;
    } // fused copy out barrier
    } // fused copy out if
  } // omp parallel
 } // run() function body

};
*/

/*********************************************************************************************************/

template<class Scalar,
         class LocalOrdinal,
         class GlobalOrdinal,
         class LocalOrdinalViewType>
void mult_A_B_newmatrix_LowThreadGustavsonKernel(CrsMatrixStruct<Scalar, LocalOrdinal, GlobalOrdinal, Kokkos::Compat::KokkosOpenMPWrapperNode>& Aview,
                                                 CrsMatrixStruct<Scalar, LocalOrdinal, GlobalOrdinal, Kokkos::Compat::KokkosOpenMPWrapperNode>& Bview,
                                                 const LocalOrdinalViewType & targetMapToOrigRow,
                                                 const LocalOrdinalViewType & targetMapToImportRow,
                                                 const LocalOrdinalViewType & Bcol2Ccol,
                                                 const LocalOrdinalViewType & Icol2Ccol,
                                                 CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Kokkos::Compat::KokkosOpenMPWrapperNode>& C,
                                                 Teuchos::RCP<const Import<LocalOrdinal,GlobalOrdinal,Kokkos::Compat::KokkosOpenMPWrapperNode> > Cimport,
                                                 const std::string& label,
                                                 const Teuchos::RCP<Teuchos::ParameterList>& params) {
  constexpr bool COPY_OUT = true;

#ifdef HAVE_TPETRA_MMM_TIMINGS
  std::string prefix_mmm = std::string("TpetraExt ") + label + std::string(": ");
  using Teuchos::TimeMonitor;
  #if ENABLE_NESTED == 1
  std::string ltg_id = COPY_OUT ? "Nested:LTGFusedCopyOut" : "Nested:LTGExplicitCopyOut";
  #else
  std::string ltg_id = COPY_OUT ? "LTGFusedCopyOut" : "LTGExplicitCopyOut";
  #endif
  Teuchos::RCP<Teuchos::TimeMonitor> MM = rcp(new TimeMonitor(*TimeMonitor::getNewTimer(prefix_mmm + std::string("MMM Newmatrix ") + ltg_id)));
#endif

  // Get my node / thread info (right from openmp)
  size_t thread_max =  Kokkos::Compat::KokkosOpenMPWrapperNode::execution_space::concurrency();
  if(!params.is_null()) {
    if(params->isParameter("openmp: ltg thread max"))
      thread_max = std::max((size_t)1,std::min(thread_max,params->get("openmp: ltg thread max",thread_max)));
  }

  typedef LTGFusedCopyFunctor<Scalar, LocalOrdinal, GlobalOrdinal, LocalOrdinalViewType, COPY_OUT> ltg_copy_fused_copy_functor_t;
  typedef typename ltg_copy_fused_copy_functor_t::range_type range_type;
  ltg_copy_fused_copy_functor_t ltg_copy_fused_copy_functor(Aview,
                                                            Bview,
                                                            targetMapToOrigRow,
                                                            targetMapToImportRow,
                                                            Bcol2Ccol,
                                                            Icol2Ccol,
                                                            C,
                                                            thread_max,
                                                            params);
//  Kokkos::parallel_for("MMM::LTG::NewMatrix::ThreadLocal",range_type(0, thread_max).set_chunk_size(1),[=](const size_t tid)

//  Kokkos::parallel_for(MMM::LTG::NewMatrix::ThreadLocal,
//                       range_type(0,ltg_copy_fused_copy_functor.thread_max).set_chunk_size(1),
//                       ltg_copy_fused_copy_functor);
  ltg_copy_fused_copy_functor.run();
  MM = Teuchos::null;
/*
  if (! ltg_copy_fused_copy_functor.COPY_OUT) {

    #ifdef HAVE_TPETRA_MMM_TIMINGS
    Teuchos::RCP<Teuchos::TimeMonitor> t = rcp(new TimeMonitor (*TimeMonitor::getNewTimer(prefix_mmm + std::string("MMM Newmatrix LTGExplicitCopyOut: CopyOut"))));
    #endif
    // Do the copy out
    copy_out_from_thread_memory(ltg_copy_fused_copy_functor.tl_rowptr,
                                ltg_copy_fused_copy_functor.tl_colind,
                                ltg_copy_fused_copy_functor.tl_values,
                                ltg_copy_fused_copy_functor.m,
                                ltg_copy_fused_copy_functor.thread_chunk,
                                ltg_copy_fused_copy_functor.row_mapC,
                                ltg_copy_fused_copy_functor.entriesC,
                                ltg_copy_fused_copy_functor.valuesC);
  }
*/
    // Sort
    constexpr bool SORT = false;
    if (SORT && (params.is_null() || params->get("sort entries",true))) {
      #ifdef HAVE_TPETRA_MMM_TIMINGS
      Teuchos::RCP<Teuchos::TimeMonitor> t = rcp(new TimeMonitor (*TimeMonitor::getNewTimer(prefix_mmm + std::string("MMM Newmatrix OpenMPSort"))));
      #endif
      Import_Util::sortCrsEntries(ltg_copy_fused_copy_functor.row_mapC,
                                  ltg_copy_fused_copy_functor.entriesC,
                                  ltg_copy_fused_copy_functor.valuesC);
    }


    // set values
    {
      #ifdef HAVE_TPETRA_MMM_TIMINGS
      Teuchos::RCP<Teuchos::TimeMonitor> t = rcp(new TimeMonitor (*TimeMonitor::getNewTimer(prefix_mmm + std::string("MMM Newmatrix C.setAllValues"))));
      #endif
      C.setAllValues(ltg_copy_fused_copy_functor.row_mapC,
                     ltg_copy_fused_copy_functor.entriesC,
                     ltg_copy_fused_copy_functor.valuesC);

    }
}

/*********************************************************************************************************/
template<class Scalar,
         class LocalOrdinal,
         class GlobalOrdinal,
         class LocalOrdinalViewType>
void mult_A_B_reuse_LowThreadGustavsonKernel(CrsMatrixStruct<Scalar, LocalOrdinal, GlobalOrdinal, Kokkos::Compat::KokkosOpenMPWrapperNode>& Aview,
                                                 CrsMatrixStruct<Scalar, LocalOrdinal, GlobalOrdinal, Kokkos::Compat::KokkosOpenMPWrapperNode>& Bview,
                                                 const LocalOrdinalViewType & targetMapToOrigRow,
                                                 const LocalOrdinalViewType & targetMapToImportRow,
                                                 const LocalOrdinalViewType & Bcol2Ccol,
                                                 const LocalOrdinalViewType & Icol2Ccol,
                                                 CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Kokkos::Compat::KokkosOpenMPWrapperNode>& C,
                                                 Teuchos::RCP<const Import<LocalOrdinal,GlobalOrdinal,Kokkos::Compat::KokkosOpenMPWrapperNode> > Cimport,
                                                 const std::string& label,
                                                 const Teuchos::RCP<Teuchos::ParameterList>& params) {
#ifdef HAVE_TPETRA_MMM_TIMINGS
  std::string prefix_mmm = std::string("TpetraExt ") + label + std::string(": ");
  using Teuchos::TimeMonitor;
  Teuchos::RCP<Teuchos::TimeMonitor> MM = rcp(new TimeMonitor(*TimeMonitor::getNewTimer(prefix_mmm + std::string("MMM Reuse LTGCore"))));
#endif

  using Teuchos::Array;
  using Teuchos::ArrayRCP;
  using Teuchos::ArrayView;
  using Teuchos::RCP;
  using Teuchos::rcp;

  // Lots and lots of typedefs
  typedef typename Kokkos::Compat::KokkosOpenMPWrapperNode Node;
  typedef typename Tpetra::CrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>::local_matrix_type KCRS;
  //  typedef typename KCRS::device_type device_t;
  typedef typename KCRS::StaticCrsGraphType graph_t;
  typedef typename graph_t::row_map_type::const_type c_lno_view_t;
  typedef typename graph_t::entries_type::const_type c_lno_nnz_view_t;
  typedef typename KCRS::values_type::non_const_type scalar_view_t;

  typedef Scalar            SC;
  typedef LocalOrdinal      LO;
  typedef GlobalOrdinal     GO;
  typedef Node              NO;
  typedef Map<LO,GO,NO>                     map_type;

  // NOTE (mfh 15 Sep 2017) This is specifically only for
  // execution_space = Kokkos::OpenMP, so we neither need nor want
  // KOKKOS_LAMBDA (with its mandatory __device__ marking).
  typedef NO::execution_space execution_space;
  typedef Kokkos::RangePolicy<execution_space, size_t> range_type;

  // All of the invalid guys
  const LO LO_INVALID = Teuchos::OrdinalTraits<LO>::invalid();
  const SC SC_ZERO = Teuchos::ScalarTraits<Scalar>::zero();
  const size_t INVALID = Teuchos::OrdinalTraits<size_t>::invalid();

  // Grab the  Kokkos::SparseCrsMatrices & inner stuff
  const KCRS & Amat = Aview.origMatrix->getLocalMatrix();
  const KCRS & Bmat = Bview.origMatrix->getLocalMatrix();
  const KCRS & Cmat = C.getLocalMatrix();

  c_lno_view_t Arowptr = Amat.graph.row_map, Browptr = Bmat.graph.row_map, Crowptr = Cmat.graph.row_map;
  const c_lno_nnz_view_t Acolind = Amat.graph.entries, Bcolind = Bmat.graph.entries, Ccolind = Cmat.graph.entries;
  const scalar_view_t Avals = Amat.values, Bvals = Bmat.values;
  scalar_view_t Cvals = Cmat.values;

  c_lno_view_t  Irowptr;
  c_lno_nnz_view_t  Icolind;
  scalar_view_t  Ivals;
  if(!Bview.importMatrix.is_null()) {
    Irowptr = Bview.importMatrix->getLocalMatrix().graph.row_map;
    Icolind = Bview.importMatrix->getLocalMatrix().graph.entries;
    Ivals   = Bview.importMatrix->getLocalMatrix().values;
  }

  // Sizes
  RCP<const map_type> Ccolmap = C.getColMap();
  size_t m = Aview.origMatrix->getNodeNumRows();
  size_t n = Ccolmap->getNodeNumElements();

  // Get my node / thread info (right from openmp or parameter list)
  size_t thread_max =  Kokkos::Compat::KokkosOpenMPWrapperNode::execution_space::concurrency();
  if(!params.is_null()) {
    if(params->isParameter("openmp: ltg thread max"))
      thread_max = std::max((size_t)1,std::min(thread_max,params->get("openmp: ltg thread max",thread_max)));
  }

  double thread_chunk = (double)(m) / thread_max;

  // Run chunks of the matrix independently
  Kokkos::parallel_for("MMM::LTG::Reuse::ThreadLocal",range_type(0, thread_max).set_chunk_size(1),[=](const size_t tid)
    {
      // Thread coordination stuff
      size_t my_thread_start =  tid * thread_chunk;
      size_t my_thread_stop  = tid == thread_max-1 ? m : (tid+1)*thread_chunk;

      // Allocations
      std::vector<size_t> c_status(n,INVALID);

      // For each row of A/C
      size_t CSR_ip = 0, OLD_ip = 0;
      for (size_t i = my_thread_start; i < my_thread_stop; i++) {
        // First fill the c_status array w/ locations where we're allowed to
        // generate nonzeros for this row
        OLD_ip = Crowptr(i);
        CSR_ip = Crowptr(i+1);
        for (size_t k = OLD_ip; k < CSR_ip; k++) {
          c_status[Ccolind(k)] = k;
          // Reset values in the row of C
          Cvals(k) = SC_ZERO;
        }

        for (size_t k = Arowptr(i); k < Arowptr(i+1); k++) {
          LO Aik  = Acolind(k);
          const SC Aval = Avals(k);
          if (Aval == SC_ZERO)
            continue;

          if (targetMapToOrigRow(Aik) != LO_INVALID) {
            // Local matrix
            size_t Bk = Teuchos::as<size_t>(targetMapToOrigRow(Aik));

            for (size_t j = Browptr(Bk); j < Browptr(Bk+1); ++j) {
              LO Bkj = Bcolind(j);
              LO Cij = Bcol2Ccol(Bkj);

              TEUCHOS_TEST_FOR_EXCEPTION(c_status[Cij] < OLD_ip || c_status[Cij] >= CSR_ip,
                                         std::runtime_error, "Trying to insert a new entry (" << i << "," << Cij << ") into a static graph " <<
                                         "(c_status = " << c_status[Cij] << " of [" << OLD_ip << "," << CSR_ip << "))");

              Cvals(c_status[Cij]) += Aval * Bvals(j);
            }
          } else {
            // Remote matrix
            size_t Ik = Teuchos::as<size_t>(targetMapToImportRow(Aik));
            for (size_t j = Irowptr(Ik); j < Irowptr(Ik+1); ++j) {
              LO Ikj = Icolind(j);
              LO Cij = Icol2Ccol(Ikj);

              TEUCHOS_TEST_FOR_EXCEPTION(c_status[Cij] < OLD_ip || c_status[Cij] >= CSR_ip,
                                         std::runtime_error, "Trying to insert a new entry (" << i << "," << Cij << ") into a static graph " <<
                                         "(c_status = " << c_status[Cij] << " of [" << OLD_ip << "," << CSR_ip << "))");

              Cvals(c_status[Cij]) += Aval * Ivals(j);
            }
          }
        }
      }
    });

  // NOTE: No copy out or "set" of data is needed here, since we're working directly with Kokkos::Views
}

/*********************************************************************************************************/
template<class Scalar,
         class LocalOrdinal,
         class GlobalOrdinal,
         class LocalOrdinalViewType>
void jacobi_A_B_newmatrix_LowThreadGustavsonKernel(Scalar omega,
                                                   const Vector<Scalar,LocalOrdinal,GlobalOrdinal, Kokkos::Compat::KokkosOpenMPWrapperNode> & Dinv,
                                                   CrsMatrixStruct<Scalar, LocalOrdinal, GlobalOrdinal, Kokkos::Compat::KokkosOpenMPWrapperNode>& Aview,
                                                   CrsMatrixStruct<Scalar, LocalOrdinal, GlobalOrdinal, Kokkos::Compat::KokkosOpenMPWrapperNode>& Bview,
                                                   const LocalOrdinalViewType & targetMapToOrigRow,
                                                   const LocalOrdinalViewType & targetMapToImportRow,
                                                   const LocalOrdinalViewType & Bcol2Ccol,
                                                   const LocalOrdinalViewType & Icol2Ccol,
                                                   CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Kokkos::Compat::KokkosOpenMPWrapperNode>& C,
                                                   Teuchos::RCP<const Import<LocalOrdinal,GlobalOrdinal,Kokkos::Compat::KokkosOpenMPWrapperNode> > Cimport,
                                                   const std::string& label,
                                                   const Teuchos::RCP<Teuchos::ParameterList>& params) {
#ifdef HAVE_TPETRA_MMM_TIMINGS
  std::string prefix_mmm = std::string("TpetraExt ") + label + std::string(": ");
  using Teuchos::TimeMonitor;
  Teuchos::RCP<Teuchos::TimeMonitor> MM = rcp(new TimeMonitor(*TimeMonitor::getNewTimer(prefix_mmm + std::string("Jacobi Newmatrix LTGCore"))));
#endif

  using Teuchos::Array;
  using Teuchos::ArrayRCP;
  using Teuchos::ArrayView;
  using Teuchos::RCP;
  using Teuchos::rcp;

  // Lots and lots of typedefs
  typedef typename Kokkos::Compat::KokkosOpenMPWrapperNode Node;
  typedef typename Tpetra::CrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>::local_matrix_type KCRS;
  //  typedef typename KCRS::device_type device_t;
  typedef typename KCRS::StaticCrsGraphType graph_t;
  typedef typename graph_t::row_map_type::non_const_type lno_view_t;
  typedef typename graph_t::row_map_type::const_type c_lno_view_t;
  typedef typename graph_t::entries_type::non_const_type lno_nnz_view_t;
  typedef typename KCRS::values_type::non_const_type scalar_view_t;

  // Unmanaged versions of the above
  typedef UnmanagedView<lno_view_t> u_lno_view_t;
  typedef UnmanagedView<lno_nnz_view_t> u_lno_nnz_view_t;
  typedef UnmanagedView<scalar_view_t> u_scalar_view_t;

  // Jacobi-specific
  typedef typename scalar_view_t::memory_space scalar_memory_space;

  typedef Scalar            SC;
  typedef LocalOrdinal      LO;
  typedef GlobalOrdinal     GO;
  typedef Node              NO;
  typedef Map<LO,GO,NO>                     map_type;

  // NOTE (mfh 15 Sep 2017) This is specifically only for
  // execution_space = Kokkos::OpenMP, so we neither need nor want
  // KOKKOS_LAMBDA (with its mandatory __device__ marking).
  typedef NO::execution_space execution_space;
  typedef Kokkos::RangePolicy<execution_space, size_t> range_type;

  // All of the invalid guys
  const LO LO_INVALID = Teuchos::OrdinalTraits<LO>::invalid();
  const SC SC_ZERO = Teuchos::ScalarTraits<Scalar>::zero();
  const size_t INVALID = Teuchos::OrdinalTraits<size_t>::invalid();

  // Grab the  Kokkos::SparseCrsMatrices & inner stuff
  const KCRS & Amat = Aview.origMatrix->getLocalMatrix();
  const KCRS & Bmat = Bview.origMatrix->getLocalMatrix();

  c_lno_view_t Arowptr = Amat.graph.row_map, Browptr = Bmat.graph.row_map;
  const lno_nnz_view_t Acolind = Amat.graph.entries, Bcolind = Bmat.graph.entries;
  const scalar_view_t Avals = Amat.values, Bvals = Bmat.values;
  size_t b_max_nnz_per_row = Bview.origMatrix->getNodeMaxNumRowEntries();

  c_lno_view_t  Irowptr;
  lno_nnz_view_t  Icolind;
  scalar_view_t  Ivals;
  if(!Bview.importMatrix.is_null()) {
    Irowptr = Bview.importMatrix->getLocalMatrix().graph.row_map;
    Icolind = Bview.importMatrix->getLocalMatrix().graph.entries;
    Ivals   = Bview.importMatrix->getLocalMatrix().values;
    b_max_nnz_per_row = std::max(b_max_nnz_per_row,Bview.importMatrix->getNodeMaxNumRowEntries());
  }

  // Jacobi-specific inner stuff
  auto Dvals = Dinv.template getLocalView<scalar_memory_space>();

  // Sizes
  RCP<const map_type> Ccolmap = C.getColMap();
  size_t m = Aview.origMatrix->getNodeNumRows();
  size_t n = Ccolmap->getNodeNumElements();
  size_t Cest_nnz_per_row = 2*C_estimate_nnz_per_row(*Aview.origMatrix,*Bview.origMatrix);

  // Get my node / thread info (right from openmp)
  size_t thread_max =  Kokkos::Compat::KokkosOpenMPWrapperNode::execution_space::concurrency();
  if(!params.is_null()) {
    if(params->isParameter("openmp: ltg thread max"))
      thread_max = std::max((size_t)1,std::min(thread_max,params->get("openmp: ltg thread max",thread_max)));
  }

  // Thread-local memory
  Kokkos::View<u_lno_view_t*> tl_rowptr("top_rowptr",thread_max);
  Kokkos::View<u_lno_nnz_view_t*> tl_colind("top_colind",thread_max);
  Kokkos::View<u_scalar_view_t*> tl_values("top_values",thread_max);

  double thread_chunk = (double)(m) / thread_max;

  // Run chunks of the matrix independently
  Kokkos::parallel_for("Jacobi::LTG::NewMatrix::ThreadLocal",range_type(0, thread_max).set_chunk_size(1),[=](const size_t tid)
    {
      // Thread coordination stuff
      size_t my_thread_start =  tid * thread_chunk;
      size_t my_thread_stop  = tid == thread_max-1 ? m : (tid+1)*thread_chunk;
      size_t my_thread_m     = my_thread_stop - my_thread_start;

      // Size estimate
      size_t CSR_alloc = (size_t) (my_thread_m*Cest_nnz_per_row*0.75 + 100);

      // Allocations
      std::vector<size_t> c_status(n,INVALID);

      u_lno_view_t Crowptr((typename u_lno_view_t::data_type)malloc(u_lno_view_t::shmem_size(my_thread_m+1)),my_thread_m+1);
      u_lno_nnz_view_t Ccolind((typename u_lno_nnz_view_t::data_type)malloc(u_lno_nnz_view_t::shmem_size(CSR_alloc)),CSR_alloc);
      u_scalar_view_t Cvals((typename u_scalar_view_t::data_type)malloc(u_scalar_view_t::shmem_size(CSR_alloc)),CSR_alloc);

      // For each row of A/C
      size_t CSR_ip = 0, OLD_ip = 0;
      for (size_t i = my_thread_start; i < my_thread_stop; i++) {
        //        printf("CMS: row %d CSR_alloc = %d\n",(int)i,(int)CSR_alloc);fflush(stdout);
        // mfh 27 Sep 2016: m is the number of rows in the input matrix A
        // on the calling process.
        Crowptr(i-my_thread_start) = CSR_ip;
        // NOTE: Vector::getLocalView returns a rank 2 view here
        SC minusOmegaDval = -omega*Dvals(i,0);

        // Entries of B
        for (size_t j = Browptr(i); j < Browptr(i+1); j++) {
          const SC Bval = Bvals(j);
          if (Bval == SC_ZERO)
            continue;
          LO Bij = Bcolind(j);
          LO Cij = Bcol2Ccol(Bij);

          // Assume no repeated entries in B
          c_status[Cij]   = CSR_ip;
          Ccolind(CSR_ip) = Cij;
          Cvals(CSR_ip)   = Bvals[j];
          CSR_ip++;
        }

        // Entries of -omega * Dinv * A * B
        // mfh 27 Sep 2016: For each entry of A in the current row of A
        for (size_t k = Arowptr(i); k < Arowptr(i+1); k++) {
          LO Aik  = Acolind(k); // local column index of current entry of A
          const SC Aval = Avals(k);   // value of current entry of A
          if (Aval == SC_ZERO)
            continue; // skip explicitly stored zero values in A

          if (targetMapToOrigRow(Aik) != LO_INVALID) {
            // mfh 27 Sep 2016: If the entry of targetMapToOrigRow
            // corresponding to the current entry of A is populated, then
            // the corresponding row of B is in B_local (i.e., it lives on
            // the calling process).

            // Local matrix
            size_t Bk = Teuchos::as<size_t>(targetMapToOrigRow(Aik));

            // mfh 27 Sep 2016: Go through all entries in that row of B_local.
            for (size_t j = Browptr(Bk); j < Browptr(Bk+1); ++j) {
              LO Bkj = Bcolind(j);
              LO Cij = Bcol2Ccol(Bkj);

              if (c_status[Cij] == INVALID || c_status[Cij] < OLD_ip) {
                // New entry
                c_status[Cij]   = CSR_ip;
                Ccolind(CSR_ip) = Cij;
                Cvals(CSR_ip)   = minusOmegaDval*Aval*Bvals(j);
                CSR_ip++;

              } else {
                Cvals(c_status[Cij]) += minusOmegaDval*Aval*Bvals(j);
              }
            }

          } else {
            // mfh 27 Sep 2016: If the entry of targetMapToOrigRow
            // corresponding to the current entry of A NOT populated (has
            // a flag "invalid" value), then the corresponding row of B is
            // in B_local (i.e., it lives on the calling process).

            // Remote matrix
            size_t Ik = Teuchos::as<size_t>(targetMapToImportRow(Aik));
            for (size_t j = Irowptr(Ik); j < Irowptr(Ik+1); ++j) {
              LO Ikj = Icolind(j);
              LO Cij = Icol2Ccol(Ikj);

              if (c_status[Cij] == INVALID || c_status[Cij] < OLD_ip){
                // New entry
                c_status[Cij]   = CSR_ip;
                Ccolind(CSR_ip) = Cij;
                Cvals(CSR_ip)   = minusOmegaDval*Aval*Ivals(j);
                CSR_ip++;

              } else {
                Cvals(c_status[Cij]) += minusOmegaDval*Aval*Ivals(j);
              }
            }
          }
        }

        // Resize for next pass if needed
        if (i+1 < my_thread_stop && CSR_ip + std::min(n,(Arowptr(i+2)-Arowptr(i+1)+1)*b_max_nnz_per_row) > CSR_alloc) {
          CSR_alloc *= 2;
          Ccolind = u_lno_nnz_view_t((typename u_lno_nnz_view_t::data_type)realloc(Ccolind.data(),u_lno_nnz_view_t::shmem_size(CSR_alloc)),CSR_alloc);
          Cvals = u_scalar_view_t((typename u_scalar_view_t::data_type)realloc(Cvals.data(),u_scalar_view_t::shmem_size(CSR_alloc)),CSR_alloc);
        }
        OLD_ip = CSR_ip;
      }

      tl_rowptr(tid) = Crowptr;
      tl_colind(tid) = Ccolind;
      tl_values(tid) = Cvals;
      Crowptr(my_thread_m) = CSR_ip;
  });



  // Do the copy out
  lno_view_t row_mapC("non_const_lnow_row", m + 1);
  lno_nnz_view_t  entriesC;
  scalar_view_t   valuesC;
  copy_out_from_thread_memory(tl_rowptr,tl_colind,tl_values,m,thread_chunk,row_mapC,entriesC,valuesC);

  //Free the unamanged views
  for(size_t i=0; i<thread_max; i++) {
    if(tl_rowptr(i).data()) free(tl_rowptr(i).data());
    if(tl_colind(i).data()) free(tl_colind(i).data());
    if(tl_values(i).data()) free(tl_values(i).data());
  }

#ifdef HAVE_TPETRA_MMM_TIMINGS
    MM = rcp(new TimeMonitor (*TimeMonitor::getNewTimer(prefix_mmm + std::string("Jacobi Newmatrix OpenMPSort"))));
#endif
    // Sort & set values
    if (params.is_null() || params->get("sort entries",true))
      Import_Util::sortCrsEntries(row_mapC, entriesC, valuesC);
    C.setAllValues(row_mapC,entriesC,valuesC);

}



/*********************************************************************************************************/
template<class Scalar,
         class LocalOrdinal,
         class GlobalOrdinal,
         class LocalOrdinalViewType>
void jacobi_A_B_reuse_LowThreadGustavsonKernel(Scalar omega,
                                                   const Vector<Scalar,LocalOrdinal,GlobalOrdinal, Kokkos::Compat::KokkosOpenMPWrapperNode> & Dinv,
                                                   CrsMatrixStruct<Scalar, LocalOrdinal, GlobalOrdinal, Kokkos::Compat::KokkosOpenMPWrapperNode>& Aview,
                                                   CrsMatrixStruct<Scalar, LocalOrdinal, GlobalOrdinal, Kokkos::Compat::KokkosOpenMPWrapperNode>& Bview,
                                                   const LocalOrdinalViewType & targetMapToOrigRow,
                                                   const LocalOrdinalViewType & targetMapToImportRow,
                                                   const LocalOrdinalViewType & Bcol2Ccol,
                                                   const LocalOrdinalViewType & Icol2Ccol,
                                                   CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Kokkos::Compat::KokkosOpenMPWrapperNode>& C,
                                                   Teuchos::RCP<const Import<LocalOrdinal,GlobalOrdinal,Kokkos::Compat::KokkosOpenMPWrapperNode> > Cimport,
                                                   const std::string& label,
                                                   const Teuchos::RCP<Teuchos::ParameterList>& params) {
#ifdef HAVE_TPETRA_MMM_TIMINGS
  std::string prefix_mmm = std::string("TpetraExt ") + label + std::string(": ");
  using Teuchos::TimeMonitor;
  Teuchos::RCP<Teuchos::TimeMonitor> MM = rcp(new TimeMonitor(*TimeMonitor::getNewTimer(prefix_mmm + std::string("Jacobi Reuse LTGCore"))));
#endif
  using Teuchos::Array;
  using Teuchos::ArrayRCP;
  using Teuchos::ArrayView;
  using Teuchos::RCP;
  using Teuchos::rcp;

  // Lots and lots of typedefs
  typedef typename Kokkos::Compat::KokkosOpenMPWrapperNode Node;
  typedef typename Tpetra::CrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>::local_matrix_type KCRS;
  //  typedef typename KCRS::device_type device_t;
  typedef typename KCRS::StaticCrsGraphType graph_t;
  typedef typename graph_t::row_map_type::const_type c_lno_view_t;
  typedef typename graph_t::entries_type::const_type c_lno_nnz_view_t;
  typedef typename KCRS::values_type::non_const_type scalar_view_t;

  // Jacobi-specific
  typedef typename scalar_view_t::memory_space scalar_memory_space;

  typedef Scalar            SC;
  typedef LocalOrdinal      LO;
  typedef GlobalOrdinal     GO;
  typedef Node              NO;
  typedef Map<LO,GO,NO>                     map_type;

  // NOTE (mfh 15 Sep 2017) This is specifically only for
  // execution_space = Kokkos::OpenMP, so we neither need nor want
  // KOKKOS_LAMBDA (with its mandatory __device__ marking).
  typedef NO::execution_space execution_space;
  typedef Kokkos::RangePolicy<execution_space, size_t> range_type;

  // All of the invalid guys
  const LO LO_INVALID = Teuchos::OrdinalTraits<LO>::invalid();
  const SC SC_ZERO = Teuchos::ScalarTraits<Scalar>::zero();
  const size_t INVALID = Teuchos::OrdinalTraits<size_t>::invalid();

  // Grab the  Kokkos::SparseCrsMatrices & inner stuff
  const KCRS & Amat = Aview.origMatrix->getLocalMatrix();
  const KCRS & Bmat = Bview.origMatrix->getLocalMatrix();
  const KCRS & Cmat = C.getLocalMatrix();

  c_lno_view_t Arowptr = Amat.graph.row_map, Browptr = Bmat.graph.row_map, Crowptr = Cmat.graph.row_map;
  const c_lno_nnz_view_t Acolind = Amat.graph.entries, Bcolind = Bmat.graph.entries, Ccolind = Cmat.graph.entries;
  const scalar_view_t Avals = Amat.values, Bvals = Bmat.values;
  scalar_view_t Cvals = Cmat.values;

  c_lno_view_t  Irowptr;
  c_lno_nnz_view_t  Icolind;
  scalar_view_t  Ivals;
  if(!Bview.importMatrix.is_null()) {
    Irowptr = Bview.importMatrix->getLocalMatrix().graph.row_map;
    Icolind = Bview.importMatrix->getLocalMatrix().graph.entries;
    Ivals   = Bview.importMatrix->getLocalMatrix().values;
  }

  // Jacobi-specific inner stuff
  auto Dvals = Dinv.template getLocalView<scalar_memory_space>();

  // Sizes
  RCP<const map_type> Ccolmap = C.getColMap();
  size_t m = Aview.origMatrix->getNodeNumRows();
  size_t n = Ccolmap->getNodeNumElements();

  // Get my node / thread info (right from openmp or parameter list)
  size_t thread_max =  Kokkos::Compat::KokkosOpenMPWrapperNode::execution_space::concurrency();
  if(!params.is_null()) {
    if(params->isParameter("openmp: ltg thread max"))
      thread_max = std::max((size_t)1,std::min(thread_max,params->get("openmp: ltg thread max",thread_max)));
  }

  double thread_chunk = (double)(m) / thread_max;

  // Run chunks of the matrix independently
  Kokkos::parallel_for("Jacobi::LTG::Reuse::ThreadLocal",range_type(0, thread_max).set_chunk_size(1),[=](const size_t tid)
    {
      // Thread coordination stuff
      size_t my_thread_start =  tid * thread_chunk;
      size_t my_thread_stop  = tid == thread_max-1 ? m : (tid+1)*thread_chunk;

      // Allocations
      std::vector<size_t> c_status(n,INVALID);

      // For each row of A/C
      size_t CSR_ip = 0, OLD_ip = 0;
      for (size_t i = my_thread_start; i < my_thread_stop; i++) {
        // First fill the c_status array w/ locations where we're allowed to
        // generate nonzeros for this row
        OLD_ip = Crowptr(i);
        CSR_ip = Crowptr(i+1);
        // NOTE: Vector::getLocalView returns a rank 2 view here
        SC minusOmegaDval = -omega*Dvals(i,0);

        for (size_t k = OLD_ip; k < CSR_ip; k++) {
          c_status[Ccolind(k)] = k;
          // Reset values in the row of C
          Cvals(k) = SC_ZERO;
        }

        // Entries of B
        for (size_t j = Browptr(i); j < Browptr(i+1); j++) {
          const SC Bval = Bvals(j);
          if (Bval == SC_ZERO)
            continue;
          LO Bij = Bcolind(j);
          LO Cij = Bcol2Ccol(Bij);

          // Assume no repeated entries in B
          Cvals(c_status[Cij]) += Bvals(j);
          CSR_ip++;
        }


        for (size_t k = Arowptr(i); k < Arowptr(i+1); k++) {
          LO Aik  = Acolind(k);
          const SC Aval = Avals(k);
          if (Aval == SC_ZERO)
            continue;

          if (targetMapToOrigRow(Aik) != LO_INVALID) {
            // Local matrix
            size_t Bk = Teuchos::as<size_t>(targetMapToOrigRow(Aik));

            for (size_t j = Browptr(Bk); j < Browptr(Bk+1); ++j) {
              LO Bkj = Bcolind(j);
              LO Cij = Bcol2Ccol(Bkj);

              TEUCHOS_TEST_FOR_EXCEPTION(c_status[Cij] < OLD_ip || c_status[Cij] >= CSR_ip,
                                         std::runtime_error, "Trying to insert a new entry (" << i << "," << Cij << ") into a static graph " <<
                                         "(c_status = " << c_status[Cij] << " of [" << OLD_ip << "," << CSR_ip << "))");

              Cvals(c_status[Cij]) += minusOmegaDval * Aval * Bvals(j);
            }
          } else {
            // Remote matrix
            size_t Ik = Teuchos::as<size_t>(targetMapToImportRow(Aik));
            for (size_t j = Irowptr(Ik); j < Irowptr(Ik+1); ++j) {
              LO Ikj = Icolind(j);
              LO Cij = Icol2Ccol(Ikj);

              TEUCHOS_TEST_FOR_EXCEPTION(c_status[Cij] < OLD_ip || c_status[Cij] >= CSR_ip,
                                         std::runtime_error, "Trying to insert a new entry (" << i << "," << Cij << ") into a static graph " <<
                                         "(c_status = " << c_status[Cij] << " of [" << OLD_ip << "," << CSR_ip << "))");

              Cvals(c_status[Cij]) += minusOmegaDval * Aval * Ivals(j);
            }
          }
        }
      }
    });

  // NOTE: No copy out or "set" of data is needed here, since we're working directly with Kokkos::Views
}


/*********************************************************************************************************/
template<class InRowptrArrayType, class InColindArrayType, class InValsArrayType,
         class OutRowptrType, class OutColindType, class OutValsType>
void copy_out_from_thread_memory(const InRowptrArrayType & Inrowptr, const InColindArrayType &Incolind, const InValsArrayType & Invalues,
                                   size_t m, double thread_chunk,
                                   OutRowptrType & row_mapC, OutColindType &entriesC, OutValsType & valuesC ) {
  typedef OutRowptrType lno_view_t;
  typedef OutColindType lno_nnz_view_t;
  typedef OutValsType scalar_view_t;
  typedef typename lno_view_t::execution_space execution_space;
  typedef Kokkos::RangePolicy<execution_space, size_t> range_type;

  // Generate the starting nnz number per thread
  size_t thread_max =  Inrowptr.size();
  size_t c_nnz_size=0;
  lno_view_t thread_start_nnz("thread_nnz",thread_max+1);

  size_t sum = 0;
  for (size_t i=0; i < thread_max; ++i){
    thread_start_nnz(i) = sum;
    sum += Inrowptr(i)(Inrowptr(i).dimension(0)-1);
  }
  thread_start_nnz(thread_max) = sum;
  c_nnz_size = thread_start_nnz(thread_max);

  // Allocate output
  lno_nnz_view_t  entriesC_(Kokkos::ViewAllocateWithoutInitializing("entriesC"), c_nnz_size); entriesC = entriesC_;
  scalar_view_t   valuesC_(Kokkos::ViewAllocateWithoutInitializing("valuesC"), c_nnz_size);  valuesC = valuesC_;

  // Copy out
  Kokkos::parallel_for("LTG::CopyOut", range_type(0, thread_max).set_chunk_size(1),[=](const size_t tid) {
      size_t my_thread_start =  tid * thread_chunk;
      size_t my_thread_stop  = tid == thread_max-1 ? m : (tid+1)*thread_chunk;
      size_t nnz_thread_start = thread_start_nnz(tid);

      for (size_t i = my_thread_start; i < my_thread_stop; i++) {
        size_t ii = i - my_thread_start;
        // Rowptr
        row_mapC(i) = nnz_thread_start + Inrowptr(tid)(ii);
        if (i==m-1) {
          row_mapC(m) = nnz_thread_start + Inrowptr(tid)(ii+1);
        }

        // Colind / Values
        for(size_t j = Inrowptr(tid)(ii); j<Inrowptr(tid)(ii+1); j++) {
          entriesC(nnz_thread_start + j) = Incolind(tid)(j);
          valuesC(nnz_thread_start + j)  = Invalues(tid)(j);
        }
      }
    });
}//end copy_out

#endif // OpenMP


/*********************************************************************************************************/
template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node, class LocalOrdinalViewType>
void jacobi_A_B_newmatrix_MultiplyScaleAddKernel(Scalar omega,
                                                  const Vector<Scalar,LocalOrdinal,GlobalOrdinal, Node> & Dinv,
                                                  CrsMatrixStruct<Scalar, LocalOrdinal, GlobalOrdinal, Node>& Aview,
                                                  CrsMatrixStruct<Scalar, LocalOrdinal, GlobalOrdinal, Node>& Bview,
                                                  const LocalOrdinalViewType & Acol2Brow,
                                                  const LocalOrdinalViewType & Acol2Irow,
                                                  const LocalOrdinalViewType & Bcol2Ccol,
                                                  const LocalOrdinalViewType & Icol2Ccol,
                                                  CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>& C,
                                                  Teuchos::RCP<const Import<LocalOrdinal,GlobalOrdinal,Node> > Cimport,
                                                  const std::string& label,
                                                  const Teuchos::RCP<Teuchos::ParameterList>& params) {
#ifdef HAVE_TPETRA_MMM_TIMINGS
  std::string prefix_mmm = std::string("TpetraExt ") + label + std::string(": ");
  using Teuchos::TimeMonitor;
  Teuchos::RCP<Teuchos::TimeMonitor> MM = rcp(new TimeMonitor(*TimeMonitor::getNewTimer(prefix_mmm + std::string("Jacobi Reuse MSAK"))));
  Teuchos::RCP<Teuchos::TimeMonitor> MM2 = rcp(new TimeMonitor(*TimeMonitor::getNewTimer(prefix_mmm + std::string("Jacobi Reuse MSAK Multiply"))));
#endif
  typedef  CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node> Matrix_t;

  // This kernel computes (I-omega Dinv A) B the slow way (for generality's sake, you must understand)
  Teuchos::ParameterList jparams;
  if(!params.is_null()) {
    jparams = *params;
    jparams.remove("openmp: algorithm",false);
    jparams.remove("cuda: algorithm",false);
  }

  // 1) Multiply A*B
  Teuchos::RCP<Matrix_t> AB = Teuchos::rcp(new Matrix_t(C.getRowMap(),0));
  Tpetra::MMdetails::mult_A_B_newmatrix(Aview,Bview,*AB,label+std::string(" MSAK"),Teuchos::rcp(&jparams,false));

#ifdef HAVE_TPETRA_MMM_TIMINGS
MM2 = rcp(new TimeMonitor(*TimeMonitor::getNewTimer(prefix_mmm + std::string("Jacobi Reuse MSAK Scale"))));
#endif

  // 2) Scale A by Dinv
  AB->leftScale(Dinv);

#ifdef HAVE_TPETRA_MMM_TIMINGS
MM2 = rcp(new TimeMonitor(*TimeMonitor::getNewTimer(prefix_mmm + std::string("Jacobi Reuse MSAK Add"))));
#endif

  // 3) Add [-omega Dinv A] + B
  Scalar one = Teuchos::ScalarTraits<Scalar>::one();
  Tpetra::MatrixMatrix::add(one,false,*Bview.origMatrix,Scalar(-omega),false,*AB,C,AB->getDomainMap(),AB->getRangeMap(),params);

 }// jacobi_A_B_newmatrix_MultiplyScaleAddKernel



}//ExtraKernels
}//MatrixMatrix
}//Tpetra

/*
static constexpr int NUM_HARDWARE_THREADS = 2;
template<class Scalar,
         class LocalOrdinal,
         class GlobalOrdinal,
         class LocalOrdinalViewType>
void mult_A_B_newmatrix_LowThreadGustavsonKernel(CrsMatrixStruct<Scalar, LocalOrdinal, GlobalOrdinal, Kokkos::Compat::KokkosOpenMPWrapperNode>& Aview,
                                                 CrsMatrixStruct<Scalar, LocalOrdinal, GlobalOrdinal, Kokkos::Compat::KokkosOpenMPWrapperNode>& Bview,
                                                 const LocalOrdinalViewType & targetMapToOrigRow,
                                                 const LocalOrdinalViewType & targetMapToImportRow,
                                                 const LocalOrdinalViewType & Bcol2Ccol,
                                                 const LocalOrdinalViewType & Icol2Ccol,
                                                 CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Kokkos::Compat::KokkosOpenMPWrapperNode>& C,
                                                 Teuchos::RCP<const Import<LocalOrdinal,GlobalOrdinal,Kokkos::Compat::KokkosOpenMPWrapperNode> > Cimport,
                                                 const std::string& label,
                                                 const Teuchos::RCP<Teuchos::ParameterList>& params) {
#ifdef HAVE_TPETRA_MMM_TIMINGS
  std::string prefix_mmm = std::string("TpetraExt ") + label + std::string(": ");
  using Teuchos::TimeMonitor;
  Teuchos::RCP<Teuchos::TimeMonitor> MM = rcp(new TimeMonitor(*TimeMonitor::getNewTimer(prefix_mmm + std::string("MMM Newmatrix LTGCore"))));
#endif

  using Teuchos::Array;
  using Teuchos::ArrayRCP;
  using Teuchos::ArrayView;
  using Teuchos::RCP;
  using Teuchos::rcp;


  // Lots and lots of typedefs
  typedef typename Kokkos::Compat::KokkosOpenMPWrapperNode Node;
  typedef typename Tpetra::CrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>::local_matrix_type KCRS;
  //  typedef typename KCRS::device_type device_t;
  typedef typename KCRS::StaticCrsGraphType graph_t;
  typedef typename graph_t::row_map_type::non_const_type lno_view_t;
  typedef typename graph_t::row_map_type::const_type c_lno_view_t;
  typedef typename graph_t::entries_type::non_const_type lno_nnz_view_t;
  typedef typename KCRS::values_type::non_const_type scalar_view_t;

  // Unmanaged versions of the above
  typedef UnmanagedView<lno_view_t> u_lno_view_t;
  typedef UnmanagedView<lno_nnz_view_t> u_lno_nnz_view_t;
  typedef UnmanagedView<scalar_view_t> u_scalar_view_t;

  typedef Scalar            SC;
  typedef LocalOrdinal      LO;
  typedef GlobalOrdinal     GO;
  typedef Node              NO;
  typedef Map<LO,GO,NO>                     map_type;

  // NOTE (mfh 15 Sep 2017) This is specifically only for
  // execution_space = Kokkos::OpenMP, so we neither need nor want
  // KOKKOS_LAMBDA (with its mandatory __device__ marking).
  typedef NO::execution_space execution_space;
  typedef Kokkos::RangePolicy<execution_space, size_t> range_type;

  // All of the invalid guys
  const LO LO_INVALID = Teuchos::OrdinalTraits<LO>::invalid();
  const SC SC_ZERO = Teuchos::ScalarTraits<Scalar>::zero();
  const size_t INVALID = Teuchos::OrdinalTraits<size_t>::invalid();

  // Grab the  Kokkos::SparseCrsMatrices & inner stuff
  const KCRS & Amat = Aview.origMatrix->getLocalMatrix();
  const KCRS & Bmat = Bview.origMatrix->getLocalMatrix();

  c_lno_view_t Arowptr = Amat.graph.row_map, Browptr = Bmat.graph.row_map;
  const lno_nnz_view_t Acolind = Amat.graph.entries, Bcolind = Bmat.graph.entries;
  const scalar_view_t Avals = Amat.values, Bvals = Bmat.values;
  size_t b_max_nnz_per_row = Bview.origMatrix->getNodeMaxNumRowEntries();

  c_lno_view_t  Irowptr;
  lno_nnz_view_t  Icolind;
  scalar_view_t  Ivals;
  if(!Bview.importMatrix.is_null()) {
    Irowptr = Bview.importMatrix->getLocalMatrix().graph.row_map;
    Icolind = Bview.importMatrix->getLocalMatrix().graph.entries;
    Ivals   = Bview.importMatrix->getLocalMatrix().values;
    b_max_nnz_per_row = std::max(b_max_nnz_per_row,Bview.importMatrix->getNodeMaxNumRowEntries());
  }

  // Sizes
  RCP<const map_type> Ccolmap = C.getColMap();
  size_t m = Aview.origMatrix->getNodeNumRows();
  size_t n = Ccolmap->getNodeNumElements();
  size_t Cest_nnz_per_row = 2*C_estimate_nnz_per_row(*Aview.origMatrix,*Bview.origMatrix);

  // Get my node / thread info (right from openmp or parameter list)
  size_t thread_max =  Kokkos::Compat::KokkosOpenMPWrapperNode::execution_space::concurrency();
  if(!params.is_null()) {
    if(params->isParameter("openmp: ltg thread max"))
      thread_max = std::max((size_t)1,std::min(thread_max,params->get("openmp: ltg thread max",thread_max)));
  }

  // Thread-local memory
  Kokkos::View<u_lno_view_t*> tl_rowptr("top_rowptr",thread_max);
  Kokkos::View<u_lno_nnz_view_t*> tl_colind("top_colind",thread_max);
  Kokkos::View<u_scalar_view_t*> tl_values("top_values",thread_max);

  // used for final construction
  lno_view_t     row_mapC("non_const_lnow_row", m + 1);
  lno_nnz_view_t entriesC;
  scalar_view_t  valuesC;

  double thread_chunk = (double)(m) / thread_max;

  // Run chunks of the matrix independently
  Kokkos::parallel_for("MMM::LTG::NewMatrix::ThreadLocal",range_type(0, thread_max).set_chunk_size(1),[=](const size_t tid)
//  // start with a parallel region over cores (maybe tiles too)
//  #pragma omp parallel
  {
    // Each team will process this chunk of work. Could change this...
    // It would probably be better to use a construct that allows better dynamic work partitioning
    // E.g., a parallel for w/dynamic, but that makes indexing the CSR column entries harder
    // (CSR_ip assumes that blocks are in order)
    // If shared ordered list was used, blocks of rows could be
    // Thread coordination stuff
    size_t my_thread_start =  tid * thread_chunk;
    size_t my_thread_stop  = tid == thread_max-1 ? m : (tid+1)*thread_chunk;
    size_t my_thread_m     = my_thread_stop - my_thread_start;

    // Size estimate
    size_t CSR_alloc = (size_t) (my_thread_m*Cest_nnz_per_row*1.2 + 100);

    // Allocations
    std::vector<size_t> c_status(n,INVALID);

    u_lno_view_t Crowptr((typename u_lno_view_t::data_type)malloc(u_lno_view_t::shmem_size(my_thread_m+1)),my_thread_m+1);
    u_lno_nnz_view_t Ccolind((typename u_lno_nnz_view_t::data_type)malloc(u_lno_nnz_view_t::shmem_size(CSR_alloc)),CSR_alloc);
    u_scalar_view_t Cvals((typename u_scalar_view_t::data_type) malloc (u_scalar_view_t::shmem_size(CSR_alloc)),CSR_alloc);

    // For each row of A/C
    size_t CSR_ip = 0, OLD_ip = 0;

    constexpr uint32_t VAL_BLOCK_SIZE = 64;
    typedef struct team_workspace {
      SC vals[VAL_BLOCK_SIZE];
      LO idxs[VAL_BLOCK_SIZE];
    } team_workspace_t;

    #if ENABLE_NESTED == 1
    omp_lock_t team_lock;
    omp_init_lock(&team_lock);


    // start a parallel region... way up here!
    #pragma omp parallel
    {
      const int ht_tid = omp_get_thread_num();
      const int parent_tid = omp_get_ancestor_thread_num(1);
      const int my_team_size = omp_get_team_size(1);
    #endif

      #if ENABLE_BLOCKED_COLS == 1
      team_workspace_t blk;
      #endif
//    // nested parallel loop, assumes OMP_NESTED=true
//    // this is intended to allow hardware threads to parallelize across rows,
//    // but stay relatively close in terms of data from A, data from B will still
//    // bounce around..

      for (size_t i = my_thread_start; i < my_thread_stop; ++i) {
        // mfh 27 Sep 2016: m is the number of rows in the input matrix A
        // on the calling process.
        #if ENABLE_NESTED == 1
        if (ht_tid == 0)
        {
          Crowptr(i-my_thread_start) = CSR_ip;
        }
        #else
        Crowptr(i-my_thread_start) = CSR_ip;
        #endif


        // mfh 27 Sep 2016: For each entry of A in the current row of A

        // parallelize this loop
        #if ENABLE_NESTED == 1
        #pragma omp for schedule(dynamic,1)
        #endif
        for (size_t k = Arowptr(i); k < Arowptr(i+1); k++) {
          LO Aik  = Acolind(k); // local column index of current entry of A
          const SC Aval = Avals(k);   // value of current entry of A
          if (Aval == SC_ZERO)
            continue; // skip explicitly stored zero values in A

          if (targetMapToOrigRow(Aik) != LO_INVALID) {
            // mfh 27 Sep 2016: If the entry of targetMapToOrigRow
            // corresponding to the current entry of A is populated, then
            // the corresponding row of B is in B_local (i.e., it lives on
            // the calling process).

            // Local matrix
            const size_t Bk = Teuchos::as<size_t>(targetMapToOrigRow(Aik));

            #if ENABLE_BLOCKED_COLS == 1
            uint32_t blk_idx = 0;
            #endif

            // mfh 27 Sep 2016: Go through all entries in that row of B_local.
            for (size_t j = Browptr(Bk); j < Browptr(Bk+1); ++j, ++blk_idx) {
              const LO Bkj = Bcolind(j);
              const LO Cij = Bcol2Ccol(Bkj);

              #if ENABLE_BLOCKED_COLS == 1
              if (c_status[Cij] == INVALID || c_status[Cij] < OLD_ip) {

                // New entry
                #if ENABLE_NESTED == 1
                  omp_set_lock(&team_lock);
                  if (c_status[Cij] == INVALID || c_status[Cij] < OLD_ip)
                  {
                    c_status[Cij]   = CSR_ip;
                    CSR_ip++;
                    Ccolind(c_status[Cij]) = Cij;
                  }
                  omp_unset_lock(&team_lock);
                #else
                  c_status[Cij]   = CSR_ip;
                  CSR_ip++;
                  Ccolind(c_status[Cij]) = Cij;
                #endif
              }

              // check if we exhaust the block size
              if (blk_idx >= VAL_BLOCK_SIZE) {
                for (uint32_t b = 0; b < VAL_BLOCK_SIZE; ++b) {
                  blk.vals[b] = Aik * blk.vals[b];
                }
                for (uint32_t b = 0; b < VAL_BLOCK_SIZE; ++b) {
                  #if ENABLE_NESTED == 1 || ENABLE_FORCE_ATOMIC_ADD == 1
                   Kokkos::atomic_fetch_add( &Cvals(c_status[(blk.idxs[b])]), blk.vals[b]);
                  #else
                   Cvals(c_status[(blk.idxs[b])]) += blk.vals[b];
                  #endif
                }
                blk_idx=0;
              } else {
                blk.vals[blk_idx] = Bvals(j);
                blk.idxs[blk_idx] = Cij;
              }
              #else
              if (c_status[Cij] == INVALID || c_status[Cij] < OLD_ip) {
                // New entry
                #if ENABLE_NESTED == 1
                  omp_set_lock(&team_lock);
                  if (c_status[Cij] == INVALID || c_status[Cij] < OLD_ip)
                  {
                    c_status[Cij]   = CSR_ip;
                    Cvals(CSR_ip) = Aval*Bvals(j);
                    CSR_ip++;
                    Ccolind(c_status[Cij]) = Cij;
                  }
                  omp_unset_lock(&team_lock);
                #else
                  c_status[Cij]   = CSR_ip;
                  Cvals(CSR_ip) = Aval*Bvals(j);
                  CSR_ip++;
                  Ccolind(c_status[Cij]) = Cij;
                #endif
              } else {
                Cvals(c_status[Cij]) += Aval*Bvals(j);
              }
              #endif
            }

            // process the blocks
            #if ENABLE_BLOCKED_COLS == 1
            for (uint32_t b = 0; b < blk_idx; ++b) {
              blk.vals[b] = Aik * blk.idxs[b];
            }

            for (uint32_t b = 0; b < blk_idx; ++b) {
              #if ENABLE_NESTED == 1 || ENABLE_FORCE_ATOMIC_ADD == 1
               Kokkos::atomic_fetch_add( &Cvals(c_status[(blk.idxs[b])]), blk.vals[b]);
              #else
               Cvals(c_status[(blk.idxs[b])]) += blk.vals[b];
              #endif
            }
            #endif

          } else {
            if (WARN_ON_REALLOC != 0) {
              std::stringstream ss;
              ss << "Thread[" << omp_get_thread_num() << "] Hit the branch!" << std::endl;
              std::cerr << ss.str ();
            }

            // mfh 27 Sep 2016: If the entry of targetMapToOrigRow
            // corresponding to the current entry of A NOT populated (has
            // a flag "invalid" value), then the corresponding row of B is
            // in B_local (i.e., it lives on the calling process).

            // Remote matrix
            size_t Ik = Teuchos::as<size_t>(targetMapToImportRow(Aik));
            for (size_t j = Irowptr(Ik); j < Irowptr(Ik+1); ++j) {
              LO Ikj = Icolind(j);
              LO Cij = Icol2Ccol(Ikj);

              if (c_status[Cij] == INVALID || c_status[Cij] < OLD_ip){
                // New entry
                c_status[Cij]   = CSR_ip;
                Ccolind(CSR_ip) = Cij;
                Cvals(CSR_ip)   = Aval*Ivals(j);
                CSR_ip++;

              } else {
                Cvals(c_status[Cij]) += Aval*Ivals(j);
              }
            }
          }
        }

        // Resize for next pass if needed
        if (i+1 < my_thread_stop && CSR_ip + std::min(n,(Arowptr(i+2)-Arowptr(i+1))*b_max_nnz_per_row) > CSR_alloc) {

          if (WARN_ON_REALLOC != 0) {
            std::stringstream ss;
            ss << "Thread[" << omp_get_thread_num() << "] Realloc! CSR_alloc = " << CSR_alloc << ", NEW = " << CSR_alloc*2 << std::endl;
            std::cerr << ss.str ();
          }
          CSR_alloc *= 2;
          Ccolind = u_lno_nnz_view_t((typename u_lno_nnz_view_t::data_type)realloc(Ccolind.data(),u_lno_nnz_view_t::shmem_size(CSR_alloc)),CSR_alloc);
          Cvals = u_scalar_view_t((typename u_scalar_view_t::data_type)realloc(Cvals.data(),u_scalar_view_t::shmem_size(CSR_alloc)),CSR_alloc);
        }
        // this is a critical section.. move to top?
        OLD_ip = CSR_ip;

      }
    #if ENABLE_NESTED  == 1
    // close parallel region
    }
    // destroy team lock
    omp_destroy_lock(&team_lock);
    #endif

    tl_rowptr(tid) = Crowptr;
    tl_colind(tid) = Ccolind;
    tl_values(tid) = Cvals;
    Crowptr(my_thread_m) = CSR_ip;

    // fuse the copy into the functor
    #if ENABLE_FUSED_COPY_OUT == 1
    #pragma omp barrier
    {

      //  tl_rowptr => Inrowptr,
      //  tl_colind => Incolind,
      //  tl_values => Invalues,


      // Generate the starting nnz number per thread
      // assuming low thread counts, simply replicate this data/computation. It's cheaper than synchronizing
      size_t c_nnz_size = 0;
      lno_view_t thread_start_nnz("thread_nnz", thread_max+1);

      size_t sum = 0;
      for (size_t i=0; i < thread_max; ++i){
        thread_start_nnz(i) = sum;
        sum += tl_rowptr(i)(tl_rowptr(i).dimension(0)-1);
      }
      thread_start_nnz(thread_max) = sum;
      c_nnz_size = thread_start_nnz(thread_max);

      #pragma omp single
      {
        // Allocate output
        lno_nnz_view_t entriesC_(Kokkos::ViewAllocateWithoutInitializing("entriesC"), c_nnz_size); entriesC = entriesC_;
        scalar_view_t  valuesC_(Kokkos::ViewAllocateWithoutInitializing("valuesC"), c_nnz_size);  valuesC = valuesC_;
      }

      #pragma omp barrier
      {
         // Copy out, we still know our thread limits, because we are in the same functor that did the the multiply
         size_t nnz_thread_start = thread_start_nnz(tid);

         for (size_t i = my_thread_start; i < my_thread_stop; i++) {
           size_t ii = i - my_thread_start;
           // Rowptr
           row_mapC(i) = nnz_thread_start + tl_rowptr(tid)(ii);
           if (i==m-1) {
             row_mapC(m) = nnz_thread_start + tl_rowptr(tid)(ii+1);
           }

           // Colind / Values
           for(size_t j = tl_rowptr(tid)(ii); j<tl_rowptr(tid)(ii+1); j++) {
             entriesC(nnz_thread_start + j) = tl_colind(tid)(j);
             valuesC(nnz_thread_start + j)  = tl_values(tid)(j);
           }
         }
      }
    }
    #endif
  });

#if ENABLE_FUSED_COPY_OUT == 0
  // Do the copy out
  copy_out_from_thread_memory(tl_rowptr,tl_colind,tl_values,m,thread_chunk,row_mapC,entriesC,valuesC);


  //Free the unamanged views
  for(size_t i=0; i<thread_max; i++) {
    if(tl_rowptr(i).data()) free(tl_rowptr(i).data());
    if(tl_colind(i).data()) free(tl_colind(i).data());
    if(tl_values(i).data()) free(tl_values(i).data());
  }
#endif

#ifdef HAVE_TPETRA_MMM_TIMINGS
    MM = rcp(new TimeMonitor (*TimeMonitor::getNewTimer(prefix_mmm + std::string("MMM Newmatrix OpenMPSort"))));
#endif
    // Sort & set values
//    if (params.is_null() || params->get("sort entries",true))
//      Import_Util::sortCrsEntries(row_mapC, entriesC, valuesC);
    C.setAllValues(row_mapC,entriesC,valuesC);

}
 */

#endif
