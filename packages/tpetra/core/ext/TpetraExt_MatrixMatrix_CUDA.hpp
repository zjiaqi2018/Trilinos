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

#ifndef TPETRA_MATRIXMATRIX_OPENMP_DEF_HPP
#define TPETRA_MATRIXMATRIX_OPENMP_DEF_HPP
#include "TpetraExt_MatrixMatrix_decl.hpp"
#ifdef HAVE_TPETRA_INST_CUDA
namespace Tpetra {
namespace MMdetails { 

/*********************************************************************************************************/
// AB NewMatrix Kernel wrappers (KokkosKernels/CUDA Version)
template<class Scalar,
         class LocalOrdinal,
         class GlobalOrdinal,
         class LocalOrdinalViewType>
void KernelWrappers<Scalar,LocalOrdinal,GlobalOrdinal,Kokkos::Compat::KokkosCudaWrapperNode,LocalOrdinalViewType>::mult_A_B_newmatrix_kernel_wrapper(CrsMatrixStruct<Scalar, LocalOrdinal, GlobalOrdinal, Kokkos::Compat::KokkosCudaWrapperNode>& Aview,
                                                                                               CrsMatrixStruct<Scalar, LocalOrdinal, GlobalOrdinal, Kokkos::Compat::KokkosCudaWrapperNode>& Bview,
                                                                                               const LocalOrdinalViewType & Acol2Brow,
                                                                                               const LocalOrdinalViewType & Acol2Irow,
                                                                                               const LocalOrdinalViewType & Bcol2Ccol,
                                                                                               const LocalOrdinalViewType & Icol2Ccol,          
                                                                                               CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Kokkos::Compat::KokkosCudaWrapperNode>& C,
                                                                                               Teuchos::RCP<const Import<LocalOrdinal,GlobalOrdinal,Kokkos::Compat::KokkosCudaWrapperNode> > Cimport,
                                                                                               const std::string& label,
                                                                                               const Teuchos::RCP<Teuchos::ParameterList>& params) {

#ifdef HAVE_TPETRA_MMM_TIMINGS
  std::string prefix_mmm = std::string("TpetraExt ") + label + std::string(": ");
  using Teuchos::TimeMonitor;
  Teuchos::RCP<TimeMonitor> MM = rcp(new TimeMonitor(*(TimeMonitor::getNewTimer(prefix_mmm + std::string("MMM Newmatrix CudaWrapper")))));
#endif
  // Node-specific code
  typedef Kokkos::Compat::KokkosCudaWrapperNode Node;
  std::string nodename("Cuda");

  // Lots and lots of typedefs
  using Teuchos::RCP;
  typedef typename Tpetra::CrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>::local_matrix_type KCRS;
  typedef typename KCRS::device_type device_t;
  typedef typename KCRS::StaticCrsGraphType graph_t;
  typedef typename graph_t::row_map_type::non_const_type lno_view_t;
  typedef typename graph_t::entries_type::non_const_type lno_nnz_view_t;
  typedef typename KCRS::values_type::non_const_type scalar_view_t;
  //typedef typename graph_t::row_map_type::const_type lno_view_t_const;

  // Options
  int team_work_size = 16;  // Defaults to 16 as per Deveci 12/7/16 - csiefer
  std::string myalg("SPGEMM_KK_MEMORY");
  if(!params.is_null()) {
    if(params->isParameter("cuda: algorithm"))
      myalg = params->get("cuda: algorithm",myalg);
    if(params->isParameter("cuda: team work size"))
      team_work_size = params->get("cuda: team work size",team_work_size);
  }

  // KokkosKernelsHandle
  typedef KokkosKernels::Experimental::KokkosKernelsHandle<
       typename lno_view_t::const_value_type,typename lno_nnz_view_t::const_value_type, typename scalar_view_t::const_value_type, 
       typename device_t::execution_space, typename device_t::memory_space,typename device_t::memory_space > KernelHandle;

  // Grab the  Kokkos::SparseCrsMatrices
  const KCRS & Ak = Aview.origMatrix->getLocalMatrix();
  const KCRS & Bk = Bview.origMatrix->getLocalMatrix();
  RCP<const KCRS> Bmerged;

  c_lno_view_t Arowptr = Amat.graph.row_map, Browptr = Bmat.graph.row_map;
  const lno_nnz_view_t Acolind = Amat.graph.entries, Bcolind = Bmat.graph.entries;
  const scalar_view_t Avals = Amat.values, Bvals = Bmat.values;

  c_lno_view_t  Irowptr;
  lno_nnz_view_t  Icolind;
  scalar_view_t  Ivals;
  if(!Bview.importMatrix.is_null()) {
    Irowptr = Bview.importMatrix->getLocalMatrix().graph.row_map;
    Icolind = Bview.importMatrix->getLocalMatrix().graph.entries;
    Ivals   = Bview.importMatrix->getLocalMatrix().values;
  }


  // Get the algorithm mode
  std::string alg = nodename+std::string(" algorithm");
  //  printf("DEBUG: Using kernel: %s\n",myalg.c_str());
  if(!params.is_null() && params->isParameter(alg)) myalg = params->get(alg,myalg);
  KokkosSparse::SPGEMMAlgorithm alg_enum = KokkosSparse::StringToSPGEMMAlgorithm(myalg);

  // We need to do this dance if either (a) We have Bimport or (b) We don't A's colMap is not the same as B's rowMap
  if(!Bview.importMatrix.is_null() ||
    (Bview.importMatrix.is_null() && (&*Aview.origMatrix->getGraph()->getColMap() != &*Bview.origMatrix->getGraph()->getRowMap())))
  {
    // We do have a Bimport
    // NOTE: We're going merge Borig and Bimport into a single matrix and reindex the columns *before* we multiply.
    // This option was chosen because we know we don't have any duplicate entries, so we can allocate once.

    typename scalar_view_t::const_type IkValues;
    typename lno_view_t::const_type IkRowPtrs;
    typename lno_nnz_view_t::const_type IkEntries;
    if(!Bview.importMatrix.is_null())
    {
      auto Ik = Teuchos::rcpFromRef<const KCRS>(Bview.importMatrix->getLocalMatrix());
      IkValues = Ik->values;
      IkRowPtrs = Ik->graph.row_map;
      IkEntries = Ik->graph.entries;
    }
    size_t merge_numrows =  Ak.numCols();
    lno_view_t Mrowptr("Mrowptr", merge_numrows + 1);

    const LocalOrdinal LO_INVALID = Teuchos::OrdinalTraits<LocalOrdinal>::invalid();

    // NOTE: This needs to get streamlined
    lno_nnz_view_t Acol2BrowDev("Acol2Brow", Acol2Brow.size());
    lno_nnz_view_t Acol2IrowDev("Acol2Irow", Acol2Irow.size());
    lno_nnz_view_t Bcol2CcolDev("Bcol2Ccol", Bcol2Ccol.size());
    lno_nnz_view_t Icol2CcolDev("Icol2Ccol", Icol2Ccol.size());
    {
      auto Acol2BrowHost = Kokkos::create_mirror_view(Acol2BrowDev);
      auto Acol2IrowHost = Kokkos::create_mirror_view(Acol2IrowDev);
      auto Bcol2CcolHost = Kokkos::create_mirror_view(Bcol2CcolDev);
      auto Icol2CcolHost = Kokkos::create_mirror_view(Icol2CcolDev);
      // use the raw pointers out of the Teuchos::Array's to avoid the Teuchos::Array+Kokkos+DEBUG problems
      // copy all four arrays to host Views, then copy those to device
      memcpy(Acol2BrowHost.ptr_on_device(), Acol2Brow.getRawPtr(), Acol2Brow.size() * sizeof(LocalOrdinal));
      memcpy(Acol2IrowHost.ptr_on_device(), Acol2Irow.getRawPtr(), Acol2Irow.size() * sizeof(LocalOrdinal));
      memcpy(Bcol2CcolHost.ptr_on_device(), Bcol2Ccol.getRawPtr(), Bcol2Ccol.size() * sizeof(LocalOrdinal));
      memcpy(Icol2CcolHost.ptr_on_device(), Icol2Ccol.getRawPtr(), Icol2Ccol.size() * sizeof(LocalOrdinal));
      Kokkos::deep_copy(Acol2BrowDev, Acol2BrowHost);
      Kokkos::deep_copy(Acol2IrowDev, Acol2IrowHost);
      Kokkos::deep_copy(Bcol2CcolDev, Bcol2CcolHost);
      Kokkos::deep_copy(Icol2CcolDev, Icol2CcolHost);
    }
    // Use a Kokkos::parallel_scan to build the rowptr
    typedef Node::execution_space execution_space;
    typedef Kokkos::RangePolicy<execution_space, size_t> range_type;
    Kokkos::parallel_scan("Tpetra_MatrixMatrix_buildRowptrBmerged", range_type (0, merge_numrows), KOKKOS_LAMBDA(const size_t i, size_t& update, const bool final) {
        if(final)
          Mrowptr(i) = update;
        // Get the row count
        size_t ct = 0;
        if(Acol2BrowDev(i) != LO_INVALID)
          ct = Browptr(Acol2BrowDev(i) + 1) - Browptr(Acol2BrowDev(i));
        else
          ct = Irowptr(Acol2IrowDev(i) + 1) - Irowptr(Acol2IrowDev(i));
        update += ct;
        if(final && (i + 1 == merge_numrows))
          Mrowptr(i + 1) = update;
      });

    execution_space::fence();

    // Allocate nnz
    size_t merge_nnz = Mrowptr(merge_numrows);
    lno_nnz_view_t Mcolind("Mcolind", merge_nnz);
    scalar_view_t Mvalues("Mvals", merge_nnz);

    // Use a Kokkos::parallel_for to fill the rowptr/colind arrays
    typedef Kokkos::RangePolicy<execution_space, size_t> range_type;

    
    /*    MColindValuesFunctor<lno_view_t, lno_nnz_view_t, scalar_view_t>
    funct2(Mvalues, Mrowptr, Mcolind,
           Acol2BrowDev, Acol2IrowDev, Bcol2CcolDev, Icol2CcolDev,
           Bk.values, Bk.graph.row_map, Bk.graph.entries,
           IkValues, IkRowPtrs, IkEntries);

           Kokkos::parallel_for("Tpetra_MatrixMatrix_buildColindValuesBmerged", range_type (0, merge_numrows), funct2);*/
    Kokkos::parallel_for("Tpetra_MatrixMatrix_buildColindValuesBmerged", range_type (0, merge_numrows), KOKKOS_LAMBDA(const size_t i) {
        if(Acol2BrowDev(i) != LO_INVALID) {
          size_t row = Acol2BrowDev(i);
          size_t start = Browptr(row);
          for(size_t j = Mrowptr(i); j < Mrowptr(i + 1); j++) {            
            Mvalues(j) = Bvals(j - Mrowptr(i) + start);
            Mcolind(j) = Bcol2CcolDev(colind(j - Mrowptr(i) + start));
          }
        }
        else
        {
          size_t row = Acol2Irow(i);
          size_t start = Irowptr(row);
          for(size_t j = Mrowptr(i); j < Mrowptr(i + 1); j++)
          {
            Mvalues(j) = Ivals(j - Mrowptr(i) + start);
            Mcolind(j) = Icol2CcolDev(Icolind(j - Mrowptr(i) + start));
          }
        }
    execution_space::fence();
    Bmerged = Teuchos::rcp(new KCRS("CrsMatrix",merge_numrows,C.getColMap()->getNodeNumElements(),merge_nnz,Mvalues,Mrowptr,Mcolind));
  }
  else {
    // We don't have a Bimport (the easy case)
    Bmerged = Teuchos::rcpFromRef(Bk);
  }

#ifdef HAVE_TPETRA_MMM_TIMINGS
  MM = rcp(new TimeMonitor (*TimeMonitor::getNewTimer(prefix_mmm + std::string("MMM Newmatrix CudaCore"))));
#endif

  // Do the multiply on whatever we've got
  typename KernelHandle::nnz_lno_t AnumRows = Ak.numRows();
  typename KernelHandle::nnz_lno_t BnumRows = Bmerged->numRows();
  typename KernelHandle::nnz_lno_t BnumCols = Bmerged->numCols();

  lno_view_t      row_mapC ("non_const_lnow_row", AnumRows + 1);
  lno_nnz_view_t  entriesC;
  scalar_view_t   valuesC;
  KernelHandle kh;
  kh.create_spgemm_handle(alg_enum);
  kh.set_team_work_size(team_work_size);

  KokkosSparse::Experimental::spgemm_symbolic(&kh,AnumRows,BnumRows,BnumCols,Ak.graph.row_map,Ak.graph.entries,false,Bmerged->graph.row_map,Bmerged->graph.entries,false,row_mapC);

  size_t c_nnz_size = kh.get_spgemm_handle()->get_c_nnz();
  if (c_nnz_size){
    entriesC = lno_nnz_view_t (Kokkos::ViewAllocateWithoutInitializing("entriesC"), c_nnz_size);
    valuesC = scalar_view_t (Kokkos::ViewAllocateWithoutInitializing("valuesC"), c_nnz_size);
  }
  KokkosSparse::Experimental::spgemm_numeric(&kh,AnumRows,BnumRows,BnumCols,Ak.graph.row_map,Ak.graph.entries,Ak.values,false,Bmerged->graph.row_map,Bmerged->graph.entries,Bmerged->values,false,row_mapC,entriesC,valuesC);
  kh.destroy_spgemm_handle();

#ifdef HAVE_TPETRA_MMM_TIMINGS
  MM = rcp(new TimeMonitor (*TimeMonitor::getNewTimer(prefix_mmm + std::string("MMM Newmatrix CudaSort"))));
#endif

  // Sort & set values
  if (params.is_null() || params->get("sort entries",true))
    Import_Util::sortCrsEntries(row_mapC, entriesC, valuesC);
  C.setAllValues(row_mapC,entriesC,valuesC);

#ifdef HAVE_TPETRA_MMM_TIMINGS
  MM = rcp(new TimeMonitor (*TimeMonitor::getNewTimer(prefix_mmm + std::string("MMM Newmatrix CudaESFC"))));
#endif

  // Final Fillcomplete
  RCP<Teuchos::ParameterList> labelList = rcp(new Teuchos::ParameterList);
  labelList->set("Timer Label",label);
  if(!params.is_null()) labelList->set("compute global constants",params->get("compute global constants",true));
  RCP<const Export<LocalOrdinal,GlobalOrdinal,Kokkos::Compat::KokkosCudaWrapperNode> > dummyExport;
  C.expertStaticFillComplete(Bview.origMatrix->getDomainMap(), Aview.origMatrix->getRangeMap(), Cimport,dummyExport,labelList);
}


/*********************************************************************************************************/
template<class Scalar,
         class LocalOrdinal,
         class GlobalOrdinal,
         class LocalOrdinalViewType>
void KernelWrappers<Scalar,LocalOrdinal,GlobalOrdinal,Kokkos::Compat::KokkosCudaWrapperNode,LocalOrdinalViewType>::mult_A_B_reuse_kernel_wrapper(CrsMatrixStruct<Scalar, LocalOrdinal, GlobalOrdinal, Kokkos::Compat::KokkosCudaWrapperNode>& Aview,
                                                                                               CrsMatrixStruct<Scalar, LocalOrdinal, GlobalOrdinal, Kokkos::Compat::KokkosCudaWrapperNode>& Bview,
                                                                                               const LocalOrdinalViewType & Acol2Brow,
                                                                                               const LocalOrdinalViewType & Acol2Irow,
                                                                                               const LocalOrdinalViewType & Bcol2Ccol,
                                                                                               const LocalOrdinalViewType & Icol2Ccol,          
                                                                                               CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Kokkos::Compat::KokkosCudaWrapperNode>& C,
                                                                                               Teuchos::RCP<const Import<LocalOrdinal,GlobalOrdinal,Kokkos::Compat::KokkosCudaWrapperNode> > Cimport,
                                                                                               const std::string& label,
                                                                                               const Teuchos::RCP<Teuchos::ParameterList>& params) {

  // FIXME: This is a temporary placholder for a CUDA reuse kernel
  mult_A_B_newmatrix_kernel_wrapper(Aview,Bview,Acol2Brow,Acol2Irow,Bcol2Ccol,Icol2Ccol,C,Cimport,label,params);
}


}//MMdetails
}//Tpetra

#endif//CUDA

#endif
