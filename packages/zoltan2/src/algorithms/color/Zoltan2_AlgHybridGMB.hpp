#ifndef _ZOLTAN2_ALGHYBRIDGMB_HPP_
#define _ZOLTAN2_ALGHYBRIDGMB_HPP_

#include <vector>
#include <unordered_map>
#include <iostream>
#include <fstream>
#include <queue>
#include <sys/time.h>
#include <algorithm>

#include "Zoltan2_Algorithm.hpp"
#include "Zoltan2_GraphModel.hpp"
#include "Zoltan2_ColoringSolution.hpp"
#include "Zoltan2_Util.hpp"
#include "Zoltan2_TPLTraits.hpp"

#include "Tpetra_Core.hpp"
#include "Teuchos_RCP.hpp"
#include "Tpetra_Import.hpp"
#include "Tpetra_FEMultiVector.hpp"

#include "KokkosKernels_Handle.hpp"
#include "KokkosKernels_IOUtils.hpp"
#include "KokkosGraph_Distance1Color.hpp"
#include "KokkosGraph_Distance1ColorHandle.hpp"
#include <stdlib.h>

//////////////////////////////////////////////
//! \file Zoltan2_AlgHybridGMB.hpp
//! \brief A hybrid version of the framework proposed by Gebremedhin, Manne, 
//!        and Boman

namespace Zoltan2 {

template <typename Adapter>
class AlgHybridGMB : public Algorithm<Adapter>
{
  public:
  
    using lno_t = typename Adapter::lno_t;
    using gno_t = typename Adapter::gno_t;
    using offset_t = typename Adapter::offset_t;
    using scalar_t = typename Adapter::scalar_t;
    using base_adapter_t = typename Adapter::base_adapter_t;
    using map_t = Tpetra::Map<lno_t, gno_t>;
    using femv_scalar_t = int;
    using femv_t = Tpetra::FEMultiVector<femv_scalar_t, lno_t, gno_t>;
    using device_type = Kokkos::Device<Kokkos::Cuda,Kokkos::Cuda::memory_space>;
    using execution_space = Kokkos::Cuda;
    using memory_space = Kokkos::Cuda::memory_space;
    double timer() {
      struct timeval tp;
      gettimeofday(&tp, NULL);
      return ((double) (tp.tv_sec) + 1e-6 * tp.tv_usec);
    }
    
  private:

    void buildModel(modelFlag_t &flags); 

    //function to invoke KokkosKernels distance-1 coloring    
    template <class ExecutionSpace, typename TempMemorySpace, 
              typename MemorySpace>
    void colorInterior(const size_t nVtx, 
                       Kokkos::View<lno_t*, device_type> adjs_view,
                       Kokkos::View<offset_t*, device_type > offset_view, 
                       Teuchos::ArrayRCP<int> colors,
                       Teuchos::RCP<femv_t> femv,
                       bool recolor=false){
      

      using KernelHandle =  KokkosKernels::Experimental::KokkosKernelsHandle
          <size_t, lno_t, lno_t, Kokkos::Cuda, Kokkos::Cuda::memory_space, 
           Kokkos::Cuda::memory_space>;
      using lno_row_view_t = Kokkos::View<offset_t*, device_type>;
      using lno_nnz_view_t = Kokkos::View<lno_t*, device_type>;

      KernelHandle kh;

      if(recolor){
        kh.create_graph_coloring_handle(KokkosGraph::COLORING_VBBIT);
      } else {
        kh.create_graph_coloring_handle(KokkosGraph::COLORING_EB);  
      }
      //kh.create_graph_coloring_handle(KokkosGraph::COLORING_EB);
      kh.set_shmem_size(16128);
      kh.set_verbose(true);
      //kh.get_graph_coloring_handle()->set_eb_num_initial_colors(10);	
      //set the initial coloring of the kh.get_graph_coloring_handle() to be
      //the data view from the femv.
      Kokkos::View<int**, Kokkos::LayoutLeft> femvColors = femv->template getLocalView<MemorySpace>();
      Kokkos::View<int*, Tpetra::Map<>::device_type >  sv = subview(femvColors, Kokkos::ALL, 0);
      Kokkos::View<int*,device_type> color_view("Colors",sv.size());
      Kokkos::deep_copy(color_view,sv);
      kh.get_graph_coloring_handle()->set_vertex_colors(color_view);
      kh.get_graph_coloring_handle()->set_tictoc(true);
      KokkosGraph::Experimental::graph_color_symbolic<KernelHandle, lno_row_view_t, lno_nnz_view_t>
                                                     (&kh, nVtx, nVtx, offset_view, adjs_view);
       
      numColors = kh.get_graph_coloring_handle()->get_num_colors();
      Kokkos::deep_copy(sv,color_view);
      std::cout<<"\nKokkosKernels Coloring: "<<kh.get_graph_coloring_handle()->get_overall_coloring_time()<<" iterations: "<<kh.get_graph_coloring_handle()->get_num_phases()<<"\n\n";
    }
    
    double doOwnedToGhosts(RCP<const map_t> mapOwnedPlusGhosts,
                         size_t nVtx,
                         ArrayView<int> owners,
                         Kokkos::View<int*, device_type>& colors){
      //std::vector<int> sendcounts(comm->getSize(), 0);
      //std::vector<gno_t> sdispls(comm->getSize()+1,0);
      int nprocs = comm->getSize();
      int* sendcnts = new int[nprocs];
      int* recvcnts = new int[nprocs];
      for(int i = 0; i < nprocs; i++){
        sendcnts[i] = 0;
        recvcnts[i] = 0;
      }
      //loop through owners, count how many vertices we'll send to each processor
      for(size_t i=0; i < owners.size(); i++){
        if(owners[i] != comm->getRank() && owners[i] != -1) sendcnts[owners[i]]++;
      }
      int status = MPI_Alltoall(sendcnts,1,MPI_INT,recvcnts,1,MPI_INT,MPI_COMM_WORLD);
      
      int* sdispls = new int[nprocs];
      int* rdispls = new int[nprocs];
      //construct sdispls (for building sendbuf), and sum the total sendcount
      sdispls[0] = 0;
      rdispls[0] = 0;
      gno_t sendsize = 0;
      gno_t recvsize = 0;
      int* sentcount = new int[nprocs];
      for(int i = 1; i < comm->getSize()+1; i++){
        sdispls[i] = sdispls[i-1] + sendcnts[i-1];
        rdispls[i] = rdispls[i-1] + recvcnts[i-1];
        sendsize += sendcnts[i-1];
        recvsize += recvcnts[i-1];
        sentcount[i-1] = 0;
        //std::cout<<comm->getRank()<<": sending "<<sendcounts[i-1]<<" GIDs to proc "<<i-1<<"\n";
      }
      int* sendbuf = new int[sendsize];
      int* recvbuf = new int[recvsize];
      
      for(size_t i = 0; i < owners.size(); i++){
        if(owners[i] != comm->getRank() && owners[i] != -1){
          int idx = sdispls[owners[i]] + sentcount[owners[i]]++;
          sendbuf[idx] = mapOwnedPlusGhosts->getGlobalElement(i+nVtx);
        }
      }

      double comm_total = 0.0;
      double comm_temp = timer();
      status = MPI_Alltoallv(sendbuf, sendcnts, sdispls, MPI_INT, recvbuf,recvcnts,rdispls,MPI_INT,MPI_COMM_WORLD);
      comm_total += timer() - comm_temp;
      
      int* recvColors = new int[sendsize];

      for(int i = 0; i < comm->getSize(); i++){
        for(int j = rdispls[i]; j < rdispls[i+1]; j++){
          lno_t lid = mapOwnedPlusGhosts->getLocalElement(recvbuf[j]);
          recvbuf[j] = colors(lid);
        }
      }
      
      comm_temp = timer();
      status = MPI_Alltoallv(recvbuf, recvcnts,rdispls,MPI_INT, recvColors, sendcnts, sdispls,MPI_INT,MPI_COMM_WORLD);
      comm_total += timer() - comm_temp;
      
      for(int i = 0; i < sendsize; i++){
        colors(mapOwnedPlusGhosts->getLocalElement(sendbuf[i])) = recvColors[i];
        //std::cout<<comm->getRank()<<": global vert "<<sendbuf[i]<<" is now color  "<<recvColors[i]<<"\n";
      }
      
      //get max send and recv counts
      /*gno_t max_send = 0;
      gno_t max_recv = 0;
      gno_t total_send = 0;
      gno_t total_recv = 0;
      gno_t avg_send = 0;
      gno_t avg_recv = 0;

      MPI_Allreduce(&sendsize, &max_send, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);
      MPI_Allreduce(&recvsize, &max_recv, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);
      MPI_Allreduce(&sendsize, &total_send, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
      MPI_Allreduce(&recvsize, &total_recv, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
      
      avg_send = total_send/comm->getSize();
      avg_recv = total_send/comm->getSize();
      if(comm->getRank()==0){
        std::cout<<"Max_send: "<<max_send<<" Max_recv: "<<max_recv<<" Total_send: "<<total_send<<" Total_recv: "<<total_recv<<" Avg_send: "<<avg_send<<" Avg_recv: "<<avg_recv<<"\n";
      }*/
      return comm_total;
      /*std::vector<gno_t> idx(comm->getSize(), 0);
      for(int i = 0; i < comm->getSize(); i++){
        idx[i] = sdispls[i];
      }
      //construct sendbuf to send GIDs to owning processes
      std::vector<gno_t> sendbuf(sendcount,0);
      for(size_t i = 0; i < owners.size(); i++){
        if(owners[i] != comm->getRank() && owners[i] != -1){
          sendbuf[idx[owners[i]]++] = mapOwnedPlusGhosts->getGlobalElement(i+nVtx);
          //std::cout<<comm->getRank()<<": sending global vert "<<mapOwnedPlusGhosts->getGlobalElement(i+nVtx)<<"\n";
        }
      }
      
      //communicate GIDs to owners
      Teuchos::ArrayView<int> sendcounts_view = Teuchos::arrayViewFromVector(sendcounts);
      Teuchos::ArrayView<gno_t> sendbuf_view = Teuchos::arrayViewFromVector(sendbuf);
      Teuchos::ArrayRCP<gno_t> recvbuf;
      std::vector<int> recvcounts(comm->getSize(),0);
      Teuchos::ArrayView<int> recvcounts_view = Teuchos::arrayViewFromVector(recvcounts);
      double comm_total = 0.0;
      double comm_temp = timer();
      Zoltan2::AlltoAllv<gno_t>(*comm, *env, sendbuf_view, sendcounts_view, recvbuf, recvcounts_view);
      comm_total += timer() - comm_temp;
      //std::cout<<comm->getRank()<<": completed GID communication to owners\n";
      
      //replace entries in recvbuf with local color
      gno_t recvcounttotal = 0;
      std::vector<int> rdispls(comm->getSize()+1,0);
      for(size_t i = 1; i < recvcounts.size()+1; i++){
        rdispls[i] = rdispls[i-1] + recvcounts[i-1];
        recvcounttotal += recvcounts[i-1];
      }

      //set colors to send back to requesting processes
      std::vector<int> sendColors(recvcounttotal,0);
      gno_t color_len = 0;
      std::vector<int> colorSendCounts(comm->getSize(),0);
      for(int i = 0; i < comm->getSize(); i++){
        colorSendCounts[i]=rdispls[i+1] - rdispls[i];
        for(int j = rdispls[i]; j < rdispls[i+1]; j++){
          lno_t lid = mapOwnedPlusGhosts->getLocalElement(recvbuf[j]);
          sendColors[j] = colors(lid);
          //std::cout<<comm->getRank()<<": global vert "<<recvbuf[j]<<" recvd from proc "<<i<<" has color "<<colors(lid)<<"\n";
        }
      }
      //communicate colors back to requesting processes
      Teuchos::ArrayView<int> sendColors_view = Teuchos::arrayViewFromVector(sendColors);
      Teuchos::ArrayRCP<int> recvColors;
      std::vector<int> recvColorsCount(comm->getSize(),0);
      Teuchos::ArrayView<int> recvColorsCount_view = Teuchos::arrayViewFromVector(recvColorsCount);
      comm_temp = timer();
      Zoltan2::AlltoAllv<int>(*comm, *env, sendColors_view, recvcounts_view, recvColors, recvColorsCount_view);
      comm_total += timer() - comm_temp;*/
      //set colors of ghosts to newly received colors
    }
    
    RCP<const base_adapter_t> adapter;
    RCP<GraphModel<base_adapter_t> > model;
    RCP<Teuchos::ParameterList> pl;
    RCP<Environment> env;
    RCP<const Teuchos::Comm<int> > comm;
    int numColors;
    
  public:
    //constructor for the  hybrid distributed distance-1 algorithm
    AlgHybridGMB(
      const RCP<const base_adapter_t> &adapter_, 
      const RCP<Teuchos::ParameterList> &pl_,
      const RCP<Environment> &env_,
      const RCP<const Teuchos::Comm<int> > &comm_)
    : adapter(adapter_), pl(pl_), env(env_), comm(comm_) {
      std::cout<<comm->getRank()<<": inside coloring constructor\n";
      numColors = 4;
      modelFlag_t flags;
      flags.reset();
      buildModel(flags);
      std::cout<<comm->getRank()<<": done constructing coloring class\n";
    }


    //Main entry point for graph coloring
    void color( const RCP<ColoringSolution<Adapter> > &solution ) {
      std::cout<<comm->getRank()<<": inside coloring function\n";
      int rank = comm->getRank(); 
      //this will color the global graph in a manner similar to Zoltan
      
      //get vertex GIDs in a locally indexed array (stolen from Ice-Sheet 
      //interface)
      std::cout<<comm->getRank()<<": getting owned vtxIDs\n";
      ArrayView<const gno_t> vtxIDs;
      ArrayView<StridedData<lno_t, scalar_t> > vwgts;
      size_t nVtx = model->getVertexList(vtxIDs, vwgts);
      //we do not use weights at this point
      std::cout<<comm->getRank()<<": getting edge list\n";
      //get edge information from the model
      ArrayView<const gno_t> adjs;
      ArrayView<const offset_t> offsets;
      ArrayView<StridedData<lno_t, scalar_t> > ewgts;
      size_t nEdge = model->getEdgeList(adjs, offsets, ewgts);
      //again, weights are not used
      
      RCP<const map_t> mapOwned;
      RCP<const map_t> mapWithCopies;
      
      std::vector<gno_t> finalGIDs;
      std::vector<offset_t> finalOffset_vec;
      std::vector<lno_t> finalAdjs_vec;      

      lno_t nInterior = 0;
      std::vector<lno_t> reorderToLocal;
      for(size_t i = 0;  i< nVtx; i++) reorderToLocal.push_back(i);
      std::cout<<comm->getRank()<<": Setting up local datastructures\n";
      //Set up a typical local mapping here.
      std::unordered_map<gno_t,lno_t> globalToLocal;
      std::vector<gno_t> ownedPlusGhosts;
      for (gno_t i = 0; i < vtxIDs.size(); i++){
        if(vtxIDs[i] < 0) std::cout<<comm->getRank()<<": found a negative GID\n";
        globalToLocal[vtxIDs[i]] = i;
        ownedPlusGhosts.push_back(vtxIDs[i]);
      }
      gno_t nghosts = 0;
      for (int i = 0; i < adjs.size(); i++){
        if(globalToLocal.count(adjs[i]) == 0){
          //new unique ghost found
          if(adjs[i] < 0) std::cout<<comm->getRank()<<": found a negative adjacency\n";
          ownedPlusGhosts.push_back(adjs[i]);
          globalToLocal[adjs[i]] = vtxIDs.size() + nghosts;
          nghosts++;
            
        }
      }
        
      finalAdjs_vec.resize(adjs.size()); 
      for(size_t i = 0; i < finalAdjs_vec.size();i++){
        finalAdjs_vec[i] = globalToLocal[adjs[i]];
      }
      for(int i = 0; i < offsets.size(); i++) finalOffset_vec.push_back(offsets[i]);
      finalGIDs = ownedPlusGhosts;
      

      std::cout<<comm->getRank()<<": creating Tpetra Maps\n";
      Tpetra::global_size_t dummy = Teuchos::OrdinalTraits
                                             <Tpetra::global_size_t>::invalid();
      mapOwned = rcp(new map_t(dummy, vtxIDs, 0, comm));

      dummy = Teuchos::OrdinalTraits <Tpetra::global_size_t>::invalid();
      mapWithCopies = rcp(new map_t(dummy, 
                                Teuchos::arrayViewFromVector(ownedPlusGhosts),
                                0, comm)); 
                                      
      //create the FEMultiVector for the distributed communication.
      //We also use the views from this datastructure as arguments to
      //KokkosKernels coloring functions.
      std::cout<<comm->getRank()<<": creating FEMultiVector\n";
      typedef Tpetra::Import<lno_t, gno_t> import_t;
      Teuchos::RCP<import_t> importer = rcp(new import_t(mapOwned, 
                                                            mapWithCopies));
      Teuchos::RCP<femv_t> femv = rcp(new femv_t(mapOwned, 
                                                    importer, 1, true));
      //Get color array to fill
      ArrayRCP<int> colors = solution->getColorsRCP();
      for(size_t i=0; i<nVtx; i++){
        colors[i] = 0;
      } 
      
      //Create random numbers seeded on global IDs so that we don't
      //need to communicate for consistency. These numbers determine
      //which vertex gets recolored in the event of a conflict.
      //taken directly from the Zoltan coloring implementation 
      std::vector<int> rand(finalGIDs.size());
      for(size_t i = 0; i < finalGIDs.size(); i++){
        std::srand(finalGIDs[i]);
        rand[i] = std::rand();
      }

      std::vector<int> ghostOwners(finalGIDs.size() - nVtx);
      std::vector<gno_t> ghosts(finalGIDs.size() - nVtx);
      for(int i = nVtx; i < finalGIDs.size(); i++) ghosts[i-nVtx] = finalGIDs[i];
      ArrayView<int> owners = Teuchos::arrayViewFromVector(ghostOwners);
      ArrayView<const gno_t> ghostGIDs = Teuchos::arrayViewFromVector(ghosts);
      Tpetra::LookupStatus ls = mapOwned->getRemoteIndexList(ghostGIDs, owners);
      
      for(int i = 0; i < finalOffset_vec.size()-1; i++){
        std::sort(finalAdjs_vec.begin()+finalOffset_vec[i],finalAdjs_vec.begin()+finalOffset_vec[i+1]);
      }
      
      ArrayView<const offset_t> finalOffsets = Teuchos::arrayViewFromVector(finalOffset_vec);
      ArrayView<const lno_t> finalAdjs = Teuchos::arrayViewFromVector(finalAdjs_vec);
      //sort adjacencies, for KokkosKernels comparison
      /*for(int i = 0; i < 100; i++){
        comm->barrier();
        femv->switchActiveMultiVector();
        double comm_tmp = timer();
        femv->doOwnedToOwnedPlusShared(Tpetra::REPLACE);
        double comm_time = timer() - comm_tmp;
        std::cout<<comm->getRank()<<": Comm "<<i<<" time: "<<comm_time<<"\n";
      }*/
      
      // call coloring function
      hybridGMB(nVtx, nInterior, finalAdjs, finalOffsets,colors,femv,finalGIDs,rand,owners,mapWithCopies);
      
      //copy colors to the output array.
      for(int i = 0; i < colors.size(); i++){
        colors[reorderToLocal[i]] = femv->getData(0)[i];
      }
      /*for(lno_t i = 0; i < nVtx; i++){
        std::cout<<comm->getRank()<<": global vert "<< finalGIDs[i] <<" is color "<< colors[reorderToLocal[i]]<<"\n";
      }*/
     
      comm->barrier();
    }
     
    void hybridGMB(const size_t nVtx,lno_t nInterior, Teuchos::ArrayView<const lno_t> adjs, 
                   Teuchos::ArrayView<const offset_t> offsets, 
                   Teuchos::ArrayRCP<int> colors, Teuchos::RCP<femv_t> femv,
                   std::vector<gno_t> reorderGIDs,
                   std::vector<int> rand,
                   ArrayView<int> owners,
                   RCP<const map_t> mapOwnedPlusGhosts){
      std::cout<<comm->getRank()<<": inside coloring algorithm\n";
      double total_time = 0.0;
      double interior_time = 0.0;
      double comm_time = 0.0;
      double comp_time = 0.0;
      double recoloring_time = 0.0;
      double conflict_detection = 0.0;
      for(int i = nVtx; i < reorderGIDs.size(); i++){
        //ghosts[i-nVtx] = reorderGIDs[i];
        //std::cout<<comm->getRank()<<": ghosts["<<i-nVtx<<"] = "<<reorderGIDs[i]<<", Owned by proc"<<owners[i-nVtx]<<"\n";
      }
      //make views out of arrayViews
      std::cout<<comm->getRank()<<": creating Kokkos Views\n"; 
      /*Kokkos::View<offset_t*, Tpetra::Map<>::device_type> host_offsets("Host Offset view", offsets.size());
      for(int i = 0; i < offsets.size(); i++){
        host_offsets(i) = offsets[i];
      }
      Kokkos::View<lno_t*, Tpetra::Map<>::device_type> host_adjs("Host Adjacencies view", adjs.size());
      for(int i = 0; i < adjs.size(); i++){
        host_adjs(i) = adjs[i];
      }
      std::cout<<comm->getRank()<<": creating recoloring graph\n";*/
     
      Kokkos::View<offset_t*, device_type> dist_degrees("Owned+Ghost degree view",rand.size());
      typename Kokkos::View<offset_t*, device_type>::HostMirror dist_degrees_host = Kokkos::create_mirror(dist_degrees);
      for(int i = 0; i < adjs.size(); i++){
        dist_degrees_host(adjs[i])++;
      }
      for(int i = 0; i < offsets.size()-1; i++){
        dist_degrees_host(i) = offsets[i+1] - offsets[i];
      }
      Kokkos::View<offset_t*, device_type> dist_offsets("Owned+Ghost Offset view", rand.size()+1);
      typename Kokkos::View<offset_t*, device_type>::HostMirror dist_offsets_host = Kokkos::create_mirror(dist_offsets);
      dist_offsets_host(0) = 0;
      uint64_t total_adjs = 0;
      for(size_t i = 1; i < rand.size()+1; i++){
        dist_offsets_host(i) = dist_degrees_host(i-1) + dist_offsets_host(i-1);
        total_adjs+= dist_degrees_host(i-1);
      }
      Kokkos::View<lno_t*, device_type> dist_adjs("Owned+Ghost adjacency view", total_adjs);
      typename Kokkos::View<lno_t*, device_type>::HostMirror dist_adjs_host = Kokkos::create_mirror(dist_adjs);
      for(size_t i = 0; i < rand.size(); i++){
        dist_degrees_host(i) = 0;
      }
      for(int i = 0; i < adjs.size(); i++) dist_adjs_host(i) = adjs[i];
      if(comm->getSize() > 1){
        for(size_t i = 0; i < nVtx; i++){
          for(size_t j = offsets[i]; j < offsets[i+1]; j++){
            if( (size_t)adjs[j] >= nVtx){
              dist_adjs_host(dist_offsets_host(adjs[j]) + dist_degrees_host(adjs[j])) = i;
              dist_degrees_host(adjs[j])++;
            }
          }
      	}
      }
      
      std::cout<<comm->getRank()<<": writing graph adjacency list out to file\n";
      std::ofstream adj_file("graph.adj");
      for(int i = 0; i < offsets.size()-1; i++){
        for(int j = dist_offsets_host(i); j < dist_offsets_host(i+1); j++){
          adj_file<<dist_adjs_host(j)<<" ";
        }
        adj_file<<"\n";
      }
      adj_file.close();
      std::cout<<comm->getRank()<<": Done writing to file\n";
      
      std::cout<<comm->getRank()<<": copying host mirrors to device views\n";
      Kokkos::deep_copy(dist_degrees, dist_degrees_host);
      Kokkos::deep_copy(dist_offsets, dist_offsets_host);
      Kokkos::deep_copy(dist_adjs, dist_adjs_host);
      std::cout<<comm->getRank()<<": done copying to device\n";
      
      std::string kokkos_only_interior = pl->get<std::string>("Kokkos_only_interior","false");
      size_t kokkosVerts = nVtx;
      if(kokkos_only_interior == "true" && comm->getSize() != 1) {
        kokkosVerts = nInterior;
      }
      std::cout<<comm->getRank()<<": done creating views, coloring interior\n";
      interior_time = timer();
      //call the KokkosKernels coloring function with the Tpetra default spaces.
      /*this->colorInterior<Tpetra::Map<>::execution_space,
                          Tpetra::Map<>::memory_space,
                          Tpetra::Map<>::memory_space>*/
      this->colorInterior<Kokkos::Cuda, Kokkos::CudaSpace, Kokkos::CudaSpace>
                 (kokkosVerts, dist_adjs, dist_offsets, colors, femv);
      interior_time = timer() - interior_time;
      total_time = interior_time;
      comp_time = interior_time;
      //This is the Kokkos version of two queues. These will attempt to be used in parallel.
      Kokkos::View<lno_t*, device_type> recoloringQueue("recoloringQueue",nVtx);
      Kokkos::parallel_for(nVtx, KOKKOS_LAMBDA(const int& i){
        recoloringQueue(i) = -1;
      });
      Kokkos::View<lno_t*, device_type, Kokkos::MemoryTraits<Kokkos::Atomic> > recoloringQueue_atomic=recoloringQueue;
      Kokkos::View<int[1], device_type> recoloringSize("Recoloring Queue Size");
      recoloringSize(0) = 0;
      Kokkos::View<int[1], device_type, Kokkos::MemoryTraits<Kokkos::Atomic> > recoloringSize_atomic = recoloringSize; 
      Kokkos::View<int*,device_type> host_rand("randVec",rand.size());
      for(size_t i = 0; i < rand.size(); i++){
        host_rand(i) = rand[i];
      }
      Kokkos::View<gno_t*, device_type> gid_view("GIDs",reorderGIDs.size());
      for(size_t i = 0; i < reorderGIDs.size(); i++){
        gid_view(i) = reorderGIDs[i];
      }
      std::cout<<comm->getRank()<<": done creating recoloring datastructures, begin initial recoloring\n";
      //bootstrap distributed coloring, add conflicting vertices to the recoloring queue.
      if(comm->getSize() > 1){
        comm->barrier();
        Kokkos::View<int**, Kokkos::LayoutLeft> femvColors = femv->template getLocalView<memory_space>();
        Kokkos::View<int*, device_type> femv_colors = subview(femvColors, Kokkos::ALL, 0);
        //femv->switchActiveMultiVector();
        //double comm_temp = timer();
        comm_time = doOwnedToGhosts(mapOwnedPlusGhosts,nVtx, owners,femv_colors); 
        //femv->doOwnedToOwnedPlusShared(Tpetra::REPLACE);
        //comm_time = timer() - comm_temp;
        total_time += comm_time;
        //femv->switchActiveMultiVector();
        //get a subview of the colors:
        /*for(lno_t i = 0; i < nVtx; i++){
          std::cout<<comm->getRank()<<": global vert "<< reorderGIDs[i] <<" is color "<< femv_colors(i)<<"\n";
        }*/
        
        //detect conflicts from the initial coloring
        /*Kokkos::parallel_for(adjs.size(), KOKKOS_LAMBDA (const int& i){
          lno_t othervtx = 0;
          for(othervtx=0; host_offsets(othervtx) <= i; othervtx++);
          othervtx--;
          if(femv_colors(host_adjs(i)) == femv_colors(othervtx)){
            if(host_rand(host_adjs(i)) > host_rand(othervtx)){
              femv_colors(host_adjs(i)) = 0;
              recoloringSize_atomic(0)++;
            } else if(host_rand(othervtx) > host_rand(host_adjs(i))) {
              femv_colors(othervtx) = 0;
              recoloringSize_atomic(0)++;
            } else {
              if(gid_view(host_adjs(i)) >= gid_view(othervtx)){
                femv_colors(host_adjs(i)) = 0;
                recoloringSize_atomic(0)++;
              } else {
                femv_colors(othervtx) = 0;
                recoloringSize_atomic(0)++;
              }
            }
          }
        });*/
        comm->barrier();
        double temp = timer();
        Kokkos::parallel_for(nVtx, KOKKOS_LAMBDA (const int& i){
          for(offset_t j = dist_offsets(i); j < dist_offsets(i+1); j++){
            int currColor = femv_colors(i);
            int nborColor = femv_colors(dist_adjs(j));
            if(currColor == nborColor ){
              if(host_rand(i) > host_rand(dist_adjs(j))){
                femv_colors(i) = 0;
                recoloringSize_atomic(0)++;
                break;
              } else if(host_rand(dist_adjs(j)) > host_rand(i)){
                femv_colors(dist_adjs(j)) = 0;
                recoloringSize_atomic(0)++;
              } else {
                if (gid_view(i) >= gid_view(dist_adjs(j))){
                  femv_colors(i) = 0;
                  recoloringSize_atomic(0)++;
                  break;
                } else {
                  femv_colors(dist_adjs(j)) = 0;
                  recoloringSize_atomic(0)++;
                }
              }
            }
          }
        });
        //ensure the parallel_for has completed before continuing.
        Kokkos::fence();
        conflict_detection += timer() - temp;
        total_time += conflict_detection;
        comp_time += conflict_detection;
      }
      std::cout<<comm->getRank()<<": done initial recoloring, begin recoloring loop\n";
      double totalPerRound[100];
      double commPerRound[100];
      double compPerRound[100];
      double recoloringPerRound[100];
      double conflictDetectionPerRound[100];
      int vertsPerRound[100];
      bool done = false; //We're only done when all processors are done
      if(comm->getSize() == 1) done = true;
      totalPerRound[0] = interior_time + comm_time + conflict_detection;
      recoloringPerRound[0] = 0;
      commPerRound[0] = comm_time;
      compPerRound[0] = interior_time + conflict_detection;
      conflictDetectionPerRound[0] = conflict_detection;
      recoloringPerRound[0] = 0;
      vertsPerRound[0] = 0;
      int distributedRounds = 1; //this is the same across all processors
      //while the queue is not empty
      while(recoloringSize(0) > 0 || !done){
        //get a subview of the colors:
        Kokkos::View<int**, Kokkos::LayoutLeft> femvColors = femv->template getLocalView<memory_space>();
        Kokkos::View<int*, device_type> femv_colors = subview(femvColors, Kokkos::ALL, 0);
        //color everything in the recoloring queue, put everything on conflict queue
        if(distributedRounds < 100) {
          int localVertsToRecolor = 0;
          for(int i = 0; i < nVtx; i ++){
            if(femv_colors(i) == 0) localVertsToRecolor++;
          }
          vertsPerRound[distributedRounds] = localVertsToRecolor;//recoloringSize(0);
        }
        std::cout<<comm->getRank()<<": starting to recolor\n";
        comm->barrier();
        double recolor_temp = timer();
        //use KokkosKernels to recolor the conflicting vertices.  
        this->colorInterior<Kokkos::Cuda,
                            Kokkos::Cuda::memory_space,
                            Kokkos::Cuda::memory_space>
                            (femv_colors.size(),dist_adjs,dist_offsets,colors,femv,false);
        recoloringPerRound[distributedRounds] = timer() - recolor_temp;
        recoloring_time += recoloringPerRound[distributedRounds];
        total_time += recoloringPerRound[distributedRounds];
        comp_time += recoloringPerRound[distributedRounds];
        compPerRound[distributedRounds] = recoloringPerRound[distributedRounds];
        totalPerRound[distributedRounds] = recoloringPerRound[distributedRounds];
        std::cout<<comm->getRank()<<": done recoloring\n";
        /*for(lno_t i = 0; i < nVtx; i++){
          std::cout<<comm->getRank()<<": global vert "<< reorderGIDs[i] <<" is color "<< femv_colors(i)<<"\n";
        }*/
            
        recoloringSize(0) = 0;
        //communicate
        comm->barrier();
        //femv->switchActiveMultiVector();
        double comm_temp = timer();
        //femv->doOwnedToOwnedPlusShared(Tpetra::REPLACE);
        commPerRound[distributedRounds] = doOwnedToGhosts(mapOwnedPlusGhosts,nVtx, owners,femv_colors); 
        commPerRound[distributedRounds] = timer() - comm_temp;
        comm_time += commPerRound[distributedRounds];
        totalPerRound[distributedRounds] += commPerRound[distributedRounds];
        total_time += commPerRound[distributedRounds];
        //femv->switchActiveMultiVector();
        //detect conflicts in parallel. For a detected conflict,
        //reset the vertex-to-be-recolored's color to 0, in order to
        //allow KokkosKernels to recolor correctly.
        /*Kokkos::parallel_for(adjs.size(), KOKKOS_LAMBDA (const int& i){
          lno_t othervtx = 0;
          for(othervtx=0; host_offsets(othervtx) <= i; othervtx++);
          othervtx--;
          if(femv_colors(host_adjs(i)) == femv_colors(othervtx)){
            if(host_rand(host_adjs(i)) > host_rand(othervtx)){
              femv_colors(host_adjs(i)) = 0;
              recoloringSize_atomic(0)++;
            } else if(host_rand(othervtx) > host_rand(host_adjs(i))) {
              femv_colors(othervtx) = 0;
              recoloringSize_atomic(0)++;
            } else {
              if(gid_view(host_adjs(i)) >= gid_view(othervtx)){
                femv_colors(host_adjs(i)) = 0;
                recoloringSize_atomic(0)++;
              } else {
                femv_colors(othervtx) = 0;
                recoloringSize_atomic(0)++;
              }
            }
          }
        });*/
        comm->barrier();
        double detection_temp = timer();
        Kokkos::parallel_for(nVtx, KOKKOS_LAMBDA (const int& i){
          for(offset_t j = dist_offsets(i); j < dist_offsets(i+1); j++){
            int currColor = femv_colors(i);
            int nborColor = femv_colors(dist_adjs(j));
            if(currColor == nborColor ){
              if(host_rand(i) > host_rand(dist_adjs(j))){
                femv_colors(i) = 0;
                recoloringSize_atomic(0)++;
                break;
              } else if(host_rand(dist_adjs(j)) > host_rand(i)){
                femv_colors(dist_adjs(j)) = 0;
                recoloringSize_atomic(0)++;
              } else {
                if (gid_view(i) >= gid_view(dist_adjs(j))){
                  femv_colors(i) = 0;
                  recoloringSize_atomic(0)++;
                  break;
                } else {
                  femv_colors(dist_adjs(j)) = 0;
                  recoloringSize_atomic(0)++;
                }
              }
            }
          }
        });
        //For Cuda, this fence is necessary to ensure the Kokkos::parallel_for is finished
        //before continuing with the coloring. 
        Kokkos::fence();
        conflictDetectionPerRound[distributedRounds] = timer() - detection_temp;
        conflict_detection += conflictDetectionPerRound[distributedRounds];
        compPerRound[distributedRounds] += conflictDetectionPerRound[distributedRounds];
        totalPerRound[distributedRounds] += conflictDetectionPerRound[distributedRounds];
        total_time += conflictDetectionPerRound[distributedRounds];
        comp_time += conflictDetectionPerRound[distributedRounds];
        
        //do a reduction to determine if we're done
        int globalDone = 0;
        int localDone = recoloringSize(0);
        Teuchos::reduceAll<int, int>(*comm,Teuchos::REDUCE_SUM,1, &localDone, &globalDone);
        //std::cout<<comm->getRank()<<": globaldone="<<globalDone<<"\n";
        //We're only allowed to stop once everyone has no work to do.
        //collectives will hang if one process exits. 
        distributedRounds++;
        done = !globalDone;
      }
      
      std::cout<<comm->getRank()<<": done recoloring loop, computing statistics\n";
      int localBoundaryVertices = 0;
      for(int i = 0; i < nVtx; i++){
        for(int j = offsets[i]; j < offsets[i+1]; j++){
          if(adjs[j] >= nVtx){
            localBoundaryVertices++;
            break;
          }
        }
      }
        
      //print how many rounds of speculating/correcting happened (this should be the same for all ranks):
      if(comm->getRank()==0) printf("did %d rounds of distributed coloring\n", distributedRounds);
      int totalVertsPerRound[100];
      int totalBoundarySize = 0;
      double finalTotalPerRound[100];
      double maxRecoloringPerRound[100];
      double minRecoloringPerRound[100];
      double finalCommPerRound[100];
      double finalCompPerRound[100];
      double finalConflictDetectionPerRound[100];
      for(int i = 0; i < 100; i++) {
        totalVertsPerRound[i] = 0;
        finalTotalPerRound[i] = 0.0;
        maxRecoloringPerRound[i] = 0.0;
        minRecoloringPerRound[i] = 0.0;
        finalCommPerRound[i] = 0.0;
        finalCompPerRound[i] = 0.0;
        finalConflictDetectionPerRound[i] = 0.0;
      }
      Teuchos::reduceAll<int,int>(*comm, Teuchos::REDUCE_SUM,1, &localBoundaryVertices,&totalBoundarySize);
      Teuchos::reduceAll<int,int>(*comm, Teuchos::REDUCE_SUM,100,vertsPerRound,totalVertsPerRound);
      Teuchos::reduceAll<int,double>(*comm, Teuchos::REDUCE_MAX,100,totalPerRound,finalTotalPerRound);
      Teuchos::reduceAll<int,double>(*comm, Teuchos::REDUCE_MAX,100,recoloringPerRound,maxRecoloringPerRound);
      Teuchos::reduceAll<int,double>(*comm, Teuchos::REDUCE_MIN,100,recoloringPerRound,minRecoloringPerRound);
      Teuchos::reduceAll<int,double>(*comm, Teuchos::REDUCE_MAX,100,commPerRound,finalCommPerRound);
      Teuchos::reduceAll<int,double>(*comm, Teuchos::REDUCE_MAX,100,compPerRound,finalCompPerRound);
      Teuchos::reduceAll<int,double>(*comm, Teuchos::REDUCE_MAX,100,conflictDetectionPerRound, finalConflictDetectionPerRound);
      printf("Rank %d: boundary size: %d\n",comm->getRank(),localBoundaryVertices);
      for(int i = 0; i < std::min(distributedRounds,100); i++){
        printf("Rank %d: recolor %d vertices in round %d\n",comm->getRank(),vertsPerRound[i],i);
        if(comm->getRank()==0) printf("recolored %d vertices in round %d\n",totalVertsPerRound[i],i);
        if(comm->getRank()==0) printf("total time in round %d: %f\n",i,finalTotalPerRound[i]);
        if(comm->getRank()==0) printf("recoloring time in round %d: %f\n",i,maxRecoloringPerRound[i]);
        if(comm->getRank()==0) printf("min recoloring time in round %d: %f\n",i,minRecoloringPerRound[i]);
        if(comm->getRank()==0) printf("conflict detection time in round %d: %f\n",i,finalConflictDetectionPerRound[i]);
        if(comm->getRank()==0) printf("comm time in round %d: %f\n",i,finalCommPerRound[i]);
        if(comm->getRank()==0) printf("comp time in round %d: %f\n",i,finalCompPerRound[i]);
      }
      double global_total_time = 0.0;
      double global_recoloring_time=0.0;
      double global_min_recoloring_time=0.0;
      double global_conflict_detection=0.0;
      double global_comm_time=0.0;
      double global_comp_time=0.0;
      double global_interior_time = 0.0;
      Teuchos::reduceAll<int,double>(*comm, Teuchos::REDUCE_MAX,1,&total_time,&global_total_time);
      Teuchos::reduceAll<int,double>(*comm, Teuchos::REDUCE_MAX,1,&recoloring_time,&global_recoloring_time);
      Teuchos::reduceAll<int,double>(*comm, Teuchos::REDUCE_MIN,1,&recoloring_time,&global_min_recoloring_time);
      Teuchos::reduceAll<int,double>(*comm, Teuchos::REDUCE_MAX,1,&conflict_detection,&global_conflict_detection);
      Teuchos::reduceAll<int,double>(*comm, Teuchos::REDUCE_MAX,1,&comm_time,&global_comm_time);
      Teuchos::reduceAll<int,double>(*comm, Teuchos::REDUCE_MAX,1,&comp_time,&global_comp_time);
      Teuchos::reduceAll<int,double>(*comm, Teuchos::REDUCE_MAX,1,&interior_time,&global_interior_time);
      comm->barrier();
      fflush(stdout);
      if(comm->getRank()==0){
        printf("Boundary size: %d\n",totalBoundarySize);
        printf("Total Time: %f\n",global_total_time);
        printf("Interior Time: %f\n",global_interior_time);
        printf("Recoloring Time: %f\n",global_recoloring_time);
        printf("Min Recoloring Time: %f\n",global_min_recoloring_time);
        printf("Conflict Detection Time: %f\n",global_conflict_detection);
        printf("Comm Time: %f\n",global_comm_time);
        printf("Comp Time: %f\n",global_comp_time);
      }
      std::cout<<comm->getRank()<<": exiting coloring\n";
    }
};

template <typename Adapter>
void AlgHybridGMB<Adapter>::buildModel(modelFlag_t &flags){
  flags.set(REMOVE_SELF_EDGES);
    
  this->env->debug(DETAILED_STATUS, "   building graph model");
  std::cout<<comm->getRank()<<": starting to construct graph model\n";
  this->model = rcp(new GraphModel<base_adapter_t>(this->adapter, this->env,
                                                   this->comm, flags));
  std::cout<<comm->getRank()<<": done constructing graph model\n";
  this->env->debug(DETAILED_STATUS, "   graph model built");
}


}//end namespace Zoltan2
#endif
