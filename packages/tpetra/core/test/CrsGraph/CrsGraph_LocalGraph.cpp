/*
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
// ************************************************************************
// @HEADER
*/


#include "Tpetra_CrsGraph.hpp"
#include "Tpetra_Details_Behavior.hpp"
#include "Tpetra_Details_determineLocalTriangularStructure.hpp"
#include "Tpetra_TestingUtilities.hpp"
#include "Tpetra_Details_getEntryOnHost.hpp"


#include <type_traits>
#include <Kokkos_Pair.hpp>
#include <Kokkos_UnorderedMap.hpp>



namespace Tpetra {
  


  template<class LocalOrdinal, class Node>
  class StaticCrsGraphAssembler {
  public:
    //! The type of the graph's local indices.
    using local_ordinal_type = LocalOrdinal;
    //! This class' Kokkos device type.
    using device_type = typename Node::device_type;
    //! This class' Kokkos execution space.
    using execution_space = typename device_type::execution_space;

    //! The type of the part of the sparse graph on each MPI process.
    using local_graph_type = Kokkos::StaticCrsGraph<local_ordinal_type,
                                                    Kokkos::LayoutLeft,
                                                    device_type>;
    
    // Taken from the Kokkos::StaticCrsGraph
    using row_map_type = typename local_graph_type::row_map_type::non_const_type;
    using entries_type = typename local_graph_type::entries_type::non_const_type;


    StaticCrsGraphAssembler(local_ordinal_type num_rows , local_ordinal_type max_nnz_per_row):
      row_map("row_map",num_rows+1),row_count("row_count",num_rows),entries("entries",max_nnz_per_row*num_rows)
    {
      using range_type =  Kokkos::RangePolicy<execution_space, LocalOrdinal>;
    
      Kokkos::parallel_for("row_map fill (const)",range_type(0,num_rows+1), KOKKOS_LAMBDA(const LocalOrdinal i) {
          row_map(i) = i*max_nnz_per_row;
        });
    }

    
    StaticCrsGraphAssembler(local_ordinal_type num_rows , row_map_type nnz_per_row):
      row_map("row_map",num_rows+1),row_count("row_count",num_rows) {
      using range_type =  Kokkos::RangePolicy<execution_space, LocalOrdinal>;

      // Note: We've initialized row_map to zero, so we can ignore the first entrey
      Kokkos::parallel_scan("row_map fill (variable)",range_type(0,num_rows), KOKKOS_LAMBDA(const LocalOrdinal i, LocalOrdinal & update, const bool final) {
          update += nnz_per_row(i);
          if(final) {
            row_map(i+1) = update;
          }
        });

      size_t nnz = ::Tpetra::Details::getEntryOnHost(row_map,num_rows);
      Kokkos::resize(entries,nnz);
    }

    KOKKOS_INLINE_FUNCTION
    void insertLocalIndices(const local_ordinal_type row, entries_type columns) const {     
      typedef typename std::remove_reference< decltype( row_count(0) ) >::type atomic_incr_type;
      LocalOrdinal numEntries = columns.extent(0);
      const unsigned offset = row_map( row ) + Kokkos::atomic_fetch_add( & row_count( row ) , atomic_incr_type(numEntries) );
      for(LocalOrdinal i=0; i<numEntries; i++) {
       entries(offset+i) = columns(i);
      }
    }
    
    KOKKOS_INLINE_FUNCTION
    void insertLocalIndices(const local_ordinal_type row, const local_ordinal_type col) const {     
      typedef typename std::remove_reference< decltype( row_count(0) ) >::type atomic_incr_type;
      const unsigned offset = row_map( row ) + Kokkos::atomic_fetch_add( & row_count( row ) , atomic_incr_type(1) );
      entries(offset) =col;
    }

    void fillComplete() {
      // FIXME: Do a sort & merge here, reducing the entries and collapsing them appropriately.
      local_graph = local_graph_type(entries,row_map);
    }

    local_graph_type getLocalGraph() {
      return local_graph;
    }

  private:
    row_map_type row_map;
    row_map_type row_count;
    entries_type entries;
    local_graph_type local_graph;
  };


  template<class e2n_view_type, class LocalOrdinal, class Node> 
  class GraphAssemblerElementToNode {
  public:
    //! The type of the graph's local indices.
    using local_ordinal_type = LocalOrdinal;

    //! This class' Kokkos device type.
    using device_type = typename Node::device_type;
    //! This class' Kokkos execution space.
    using execution_space = typename device_type::execution_space;

    //! The type of the part of the sparse graph on each MPI process.
    using local_graph_type = Kokkos::StaticCrsGraph<local_ordinal_type,
                                                    Kokkos::LayoutLeft,
                                                    device_type>;
    
    // Taken from the Kokkos::StaticCrsGraph
    using row_map_type = typename local_graph_type::row_map_type::non_const_type;
    using entries_type = typename local_graph_type::entries_type::non_const_type;


    // Stuff for the UnorderedMap
    using key_type = Kokkos::pair<local_ordinal_type,local_ordinal_type>;
    using unordered_map_type = Kokkos::UnorderedMap< key_type, void , execution_space >;

    static local_graph_type Assemble(LocalOrdinal num_rows, const e2n_view_type & element_to_node_map) { 
      using size_type    = typename row_map_type::size_type;
      using entries_data_type = typename local_graph_type::data_type;
      using range_policy =  Kokkos::RangePolicy<execution_space, LocalOrdinal>;
      using LO = local_ordinal_type;

      row_map_type row_map("row_map",num_rows+1);
      row_map_type row_count(Kokkos::ViewAllocateWithoutInitializing("row_count"),num_rows);
      entries_type entries;
      local_graph_type local_graph;
    
      unordered_map_type node2node;
    

      // Start with an initial allocation
      size_t set_capacity = (((28ull * num_rows) / 2ull)*4ull)/3ull;
      LocalOrdinal failed_insert_count = 0 ;
      
      /* Build the node2node map, increasing capacity until we can store it all */
      do {
        // Zero the row count to restart the fill
        Kokkos::deep_copy( row_count , 0 );
        
        node2node = unordered_map_type( ( set_capacity += failed_insert_count ) );

        // May be larger that requested:
        set_capacity = node2node.capacity();
        
        Kokkos::parallel_reduce("count_fill",range_policy(0,element_to_node_map.extent(0)),KOKKOS_LAMBDA(const LO elem,LO &count) {
            // Do a symmetric fill here
            for(LO i=0; i<(int)element_to_node_map.extent(1); i++) {
              LO row = element_to_node_map(elem,i);
              for(LO j=i; j<(int)element_to_node_map.extent(1); j++) {
                LO col = element_to_node_map(elem,j);
                const key_type key = (row < col) ? Kokkos::make_pair( row, col ) : Kokkos::make_pair( col, row ) ;
                auto result = node2node.insert( key );                
                if ( result.success() ) {
                  // First time this pair was inserted
                  Kokkos::atomic_increment( & row_count( row ) );
                  if ( row != col) Kokkos::atomic_increment( & row_count( col ) );
                }
                else if ( result.failed() ) {
                  // Ran out of memory for insertion.
                  ++count ;
                }
              }
            }
          },failed_insert_count);

      } while ( failed_insert_count );
      
      /* Fill the row_map */
      LO nnz=0;
      Kokkos::parallel_scan("count_to_rowmap",range_policy(0,num_rows),KOKKOS_LAMBDA(const LO row,LO & update, const bool final) {
          if(final) row_map(row) = update;
          update+=row_count(row);
          if(final && row+1 == (LO)row_count.extent(0)) {
            row_map(row+1) = update;
          }

        },nnz);
      local_graph.row_map = row_map;

      /* Allocate Entries */
      Kokkos::resize(entries,nnz);

      /* Fill the graph */ 
      Kokkos::deep_copy( row_count , 0 );// We're going to use this to figure out who gets to put what where
      
      Kokkos::parallel_for("fill_graph",range_policy(0,node2node.capacity()),KOKKOS_LAMBDA(const LO iset) {
          typedef typename std::remove_reference< decltype( row_count(0) ) >::type atomic_incr_type;              
          if ( node2node.valid_at(iset) ) {
            const key_type key = node2node.key_at(iset) ;
            const LO row_node = key.first ;
            const LO col_node = key.second ;
            
            const LO offset = row_map( row_node ) + Kokkos::atomic_fetch_add( & row_count( row_node ) , atomic_incr_type(1) );
            entries( offset ) = col_node ;
            
            if ( col_node != row_node ) {
              const LO offset = row_map( col_node ) + Kokkos::atomic_fetch_add( & row_count( col_node ) , atomic_incr_type(1) );
              entries( offset ) = row_node ;
            }
          }
        });
      
      /* Sort the rows of the graph */
      Kokkos::parallel_for("count_fill",range_policy(0,num_rows),KOKKOS_LAMBDA(const LO row) {
          const size_type row_beg = row_map( row );
          const size_type row_end = row_map( row + 1 );
          for ( size_type i = row_beg + 1 ; i < row_end ; ++i ) {
            const entries_data_type col = entries(i);
            size_type j = i ;
            for ( ; row_beg < j && col < entries(j-1) ; --j ) {
              entries(j) = entries(j-1);
            }
            entries(j) = col ;
          }
        });
      
      
      local_graph.entries = entries;

      return local_graph;

    }// Assemble


  };//class


} // end namespace

  //
  // UNIT TESTS
  //


namespace {//anonymous

  using Teuchos::RCP;
  using Teuchos::rcp;
  using Teuchos::Comm;
  using Teuchos::REDUCE_SUM;
  using Teuchos::outArg;
     


  TEUCHOS_UNIT_TEST_TEMPLATE_3_DECL( CrsGraph, DeletedDiagonal_Constant_Single, LO, GO , Node )
  {
    using CrsGraph = Tpetra::CrsGraph<LO, GO, Node>;
    using map_type = Tpetra::Map<LO, GO, Node>;
    using execution_space = typename Node::execution_space;
    using range_type =  Kokkos::RangePolicy<execution_space, LO>;
    using local_graph_type = typename CrsGraph::local_graph_type;


    const GO INVALID = Teuchos::OrdinalTraits<GO>::invalid ();
    // get a comm
    RCP<const Comm<int> > comm = Tpetra::getDefaultComm();
    // create a Map, three rows per processor
    const size_t numLocal = 3;
    RCP<const map_type> map = rcp (new map_type (INVALID, numLocal, 0, comm));

    // Direct CrsGraph host assembly
    CrsGraph G1(map,map,3);
    for(LO i=0; i<(LO)numLocal; i++) {      
      for(LO j=0; j<(LO)numLocal; j++) {
        if(i!=j) 
          G1.insertLocalIndices(i,1,&j);
      }
    }
    G1.fillComplete();
    local_graph_type G1_local = G1.getLocalGraph();


    // SCGA 
    Tpetra::StaticCrsGraphAssembler<LO,Node> Assembler(numLocal,numLocal-1);
    Kokkos::parallel_for("LocalGraph Test",range_type(0,numLocal), KOKKOS_LAMBDA(const LO i) {
        for(LO j=0; j<(int)numLocal; j++)
          if(i!=j)
            Assembler.insertLocalIndices(i,j);
      });
    Assembler.fillComplete();
    local_graph_type G2_local = Assembler.getLocalGraph();

    // Check equivalence
    success=true;
    
    LO mismatches =0;
    Kokkos::parallel_reduce("row_map mismatch",range_type(0,numLocal+1),KOKKOS_LAMBDA(const LO i, LO& isum) {
        if(G1_local.row_map(i) != G2_local.row_map(i)) isum++;          
      },mismatches);
    if(mismatches > 0)  success=false;

    mismatches = 0;
    Kokkos::parallel_reduce("entries mismatch",range_type(0,G1_local.entries.extent(0)),KOKKOS_LAMBDA(const LO i, LO& isum) {
        if(G1_local.entries(i) != G2_local.entries(i)) isum++;          
      },mismatches);
    if(mismatches > 0)  success=false;


#if 0
    printf("G1 rowptr: ");
    for(LO i=0; i<(LO)numLocal+1; i++)
      printf("%d ",(int) G1_local.row_map(i));
    printf("\n");
    printf("G2 rowptr: ");
    for(LO i=0; i<(LO)numLocal+1; i++)
      printf("%d ",(int) G2_local.row_map(i));
    printf("\n");

    printf("G1 colind: ");
    for(LO i=0; i<(LO)G1_local.entries.extent(0); i++)
      printf("%d ",(int) G1_local.entries(i));
    printf("\n");
    printf("G2 colind: ");
    for(LO i=0; i<(LO)G2_local.entries.extent(0); i++)
      printf("%d ",(int) G2_local.entries(i));
    printf("\n");
#endif

    // All procs fail if any node fails
    int globalSuccess_int = -1;
    reduceAll( *comm, REDUCE_SUM, success ? 0 : 1, outArg(globalSuccess_int) );
    TEST_EQUALITY_CONST( globalSuccess_int, 0 );
  }
  


   TEUCHOS_UNIT_TEST_TEMPLATE_3_DECL( CrsGraph, Element2Node, LO, GO , Node )
  {
    using CrsGraph = Tpetra::CrsGraph<LO, GO, Node>;
    using map_type = Tpetra::Map<LO, GO, Node>;
    using device_type = typename Node::device_type;
    using execution_space = typename Node::execution_space;
    using range_type =  Kokkos::RangePolicy<execution_space, LO>;
    using local_graph_type = typename CrsGraph::local_graph_type;
    using el2node_type = Kokkos::View<LO**,Kokkos::LayoutLeft,device_type>;

    const GO INVALID = Teuchos::OrdinalTraits<GO>::invalid ();
    // get a comm
    RCP<const Comm<int> > comm = Tpetra::getDefaultComm();
    // create a Map, three rows per processor
    const size_t numLocal = 3;
    RCP<const map_type> map = rcp (new map_type (INVALID, numLocal, 0, comm));

    // Direct CrsGraph host assembly
    CrsGraph G1(map,map,3);
    for(LO i=0; i<(LO)numLocal; i++) {    
      int index=0;
      if(i > 0) {index=i-1; G1.insertLocalIndices(i,1,&index);}
      G1.insertLocalIndices(i,1,&i);
      if(i < (LO)numLocal-1) {index=i+1; G1.insertLocalIndices(i,1,&index);}
    }
    G1.fillComplete();
    local_graph_type G1_local = G1.getLocalGraph();


    // Build the element-to-node map (tridiagonal springs)
    el2node_type el2node("el2node",numLocal-1,2);
    Kokkos::parallel_for("e2n assembly",range_type(0,numLocal-1), KOKKOS_LAMBDA(const LO i) {        
        el2node(i,0) = i;
        el2node(i,1) = i+1;
      });
    

    // FEM Assembler
    local_graph_type G2_local = Tpetra::GraphAssemblerElementToNode<el2node_type,LO,Node>::Assemble(numLocal,el2node);

    // Check equivalence
    success=true;
    
    LO mismatches =0;
    Kokkos::parallel_reduce("row_map mismatch",range_type(0,numLocal+1),KOKKOS_LAMBDA(const LO i, LO& isum) {
        if(G1_local.row_map(i) != G2_local.row_map(i)) isum++;          
      },mismatches);
    if(mismatches > 0)  success=false;

    mismatches = 0;
    Kokkos::parallel_reduce("entries mismatch",range_type(0,G1_local.entries.extent(0)),KOKKOS_LAMBDA(const LO i, LO& isum) {
        if(G1_local.entries(i) != G2_local.entries(i)) isum++;          
      },mismatches);
    if(mismatches > 0)  success=false;


#if 0
    printf("G1 rowptr: ");
    for(LO i=0; i<(LO)numLocal+1; i++)
      printf("%d ",(int) G1_local.row_map(i));
    printf("\n");
    printf("G2 rowptr: ");
    for(LO i=0; i<(LO)numLocal+1; i++)
      printf("%d ",(int) G2_local.row_map(i));
    printf("\n");

    printf("G1 colind: ");
    for(LO i=0; i<(LO)G1_local.entries.extent(0); i++)
      printf("%d ",(int) G1_local.entries(i));
    printf("\n");
    printf("G2 colind: ");
    for(LO i=0; i<(LO)G2_local.entries.extent(0); i++)
      printf("%d ",(int) G2_local.entries(i));
    printf("\n");
#endif

    // All procs fail if any node fails
    int globalSuccess_int = -1;
    reduceAll( *comm, REDUCE_SUM, success ? 0 : 1, outArg(globalSuccess_int) );
    TEST_EQUALITY_CONST( globalSuccess_int, 0 );
  }


//
// INSTANTIATIONS
//

// Tests to build and run.  We will instantiate them over all enabled
// LocalOrdinal (LO), GlobalOrdinal (GO), and Node (NODE) types.
#define UNIT_TEST_GROUP( LO, GO, NODE ) \
  TEUCHOS_UNIT_TEST_TEMPLATE_3_INSTANT( CrsGraph, DeletedDiagonal_Constant_Single,   LO, GO, NODE ) \
  TEUCHOS_UNIT_TEST_TEMPLATE_3_INSTANT( CrsGraph, Element2Node,      LO, GO, NODE ) 

  TPETRA_ETI_MANGLING_TYPEDEFS()

  TPETRA_INSTANTIATE_LGN( UNIT_TEST_GROUP )

} // namespace (anonymous)
