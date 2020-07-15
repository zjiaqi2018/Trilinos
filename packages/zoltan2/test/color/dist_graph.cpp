/*
//@HEADER
// *****************************************************************************
//
//  HPCGraph: Graph Computation on High Performance Computing Systems
//              Copyright (2016) Sandia Corporation
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
// Questions?  Contact  George M. Slota   (gmslota@sandia.gov)
//                      Siva Rajamanickam (srajama@sandia.gov)
//                      Kamesh Madduri    (madduri@cse.psu.edu)
//
// *****************************************************************************
//@HEADER
*/

#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <assert.h>

#include "fast_map.h"
#include "dist_graph.h"
#include "util.h"

extern int procid, nprocs;
extern bool verbose, debug;

int create_graph(graph_gen_data_t *ggi, color_dist_graph_t *g)
{
  double elt = 0.0;
  if (debug) { 
    printf("Task %d create_graph() start\n", procid);
    MPI_Barrier(MPI_COMM_WORLD);
    elt = omp_get_wtime();
  }

  g->n = ggi->n;
  g->n_local = ggi->n_local;
  g->n_offset = ggi->n_offset;
  g->m = ggi->m;
  g->m_local = ggi->m_local_edges;
  g->map = (color_fast_map*)malloc(sizeof(color_fast_map));

  if (debug) printf("Task %d, n_local %lu n_offset %lu\n", 
    procid, g->n_local, g->n_offset);

  uint64_t* out_edges = (uint64_t*)malloc(g->m_local*sizeof(uint64_t));
  uint64_t* out_offsets = (uint64_t*)malloc((g->n_local+1)*sizeof(uint64_t));
  uint64_t* temp_counts = (uint64_t*)malloc(g->n_local*sizeof(uint64_t));
  if (out_edges == NULL || out_offsets == NULL || temp_counts == NULL)
   color_throw_err("create_graph(), unable to allocate graph edge storage", procid);

//#pragma omp parallel for 
  for (uint64_t i = 0; i < g->n_local+1; ++i)
    out_offsets[i] = 0;
//#pragma omp parallel for 
  for (uint64_t i = 0; i < g->n_local; ++i)
    temp_counts[i] = 0;

//#pragma omp parallel for 
  for (uint64_t i = 0; i < g->m_local*2; i+=2) {
    assert(ggi->gen_edges[i] >= g->n_offset);
//#pragma omp atomic
    ++temp_counts[ggi->gen_edges[i] - g->n_offset];
  }

  parallel_prefixsums(temp_counts, out_offsets+1, g->n_local);

//#pragma omp parallel for
  for (uint64_t i = 0; i < g->n_local; ++i)
    temp_counts[i] = out_offsets[i];

//#pragma omp parallel for
  for (uint64_t i = 0; i < g->m_local*2; i+=2) {
    int64_t index = -1;
    uint64_t src = ggi->gen_edges[i];
    uint64_t dst = ggi->gen_edges[i+1];
    assert(src < ggi->n);
    assert(dst < ggi->n);
//#pragma omp atomic capture
  { index = temp_counts[src - g->n_offset]; temp_counts[src - g->n_offset]++; }
    out_edges[index] = dst;
  }
  
  free(ggi->gen_edges);
  free(temp_counts);

  g->out_edges = out_edges;
  g->out_offsets = out_offsets;
  g->local_unmap = (uint64_t*)malloc(g->n_local*sizeof(uint64_t));
  if (g->local_unmap == NULL)
    color_throw_err("create_graph(), unable to allocate unmap", procid);

//#pragma omp parallel for
  for (uint64_t i = 0; i < g->n_local; ++i)
    g->local_unmap[i] = i + g->n_offset;

  if (debug) {
    for (uint64_t i = 0; i < g->n_local; i++){
      printf("Task %d: global vertex %lu neighbors\n",procid, g->local_unmap[i]);
      for(uint64_t j = g->out_offsets[i]; j < g->out_offsets[i+1]; j++){
        printf("%lu ",g->out_edges[j]);
      }
      printf("\n");
    }
    printf("Task %d create_graph() done: %lf (s)\n", 
      procid, omp_get_wtime() - elt);
  }

  return 0;
}

int create_graph_serial(graph_gen_data_t *ggi, color_dist_graph_t *g)
{
  double elt = 0.0;
  if (debug) { 
    printf("Task %d create_graph_serial() start\n", procid);
    MPI_Barrier(MPI_COMM_WORLD);
    elt = omp_get_wtime();
  }

  g->n = ggi->n_local;
  g->n_local = ggi->n_local;
  g->n_offset = 0;
  g->m = ggi->m_local_read;
  g->m_local = ggi->m_local_read*2;
  g->n_ghost = 0;
  g->n_total = g->n_local;
  g->map = (struct color_fast_map*)malloc(sizeof(struct color_fast_map));

  uint64_t* out_edges = (uint64_t*)malloc(g->m_local*sizeof(uint64_t));
  uint64_t* out_offsets = (uint64_t*)malloc((g->n_local+1)*sizeof(uint64_t));
  uint64_t* temp_counts = (uint64_t*)malloc(g->n_local*sizeof(uint64_t));
  if (out_edges == NULL || out_offsets == NULL || temp_counts == NULL)
  color_throw_err("create_graph_serial(), unable to allocate out edge storage\n", procid);
  
#pragma omp parallel for 
  for (uint64_t i = 0; i < g->n_local+1; ++i)
    out_offsets[i] = 0;
#pragma omp parallel for 
  for (uint64_t i = 0; i < g->n_local; ++i)
    temp_counts[i] = 0;

#pragma omp parallel for 
  for (uint64_t i = 0; i < g->m_local; i++)
#pragma omp atomic
    ++temp_counts[ggi->gen_edges[i] - g->n_offset];

  parallel_prefixsums(temp_counts, out_offsets+1, g->n_local);

#pragma omp parallel for
  for (uint64_t i = 0; i < g->n_local; ++i)
    temp_counts[i] = out_offsets[i];

#pragma omp parallel for
  for (uint64_t i = 0; i < g->m_local; i+=2) {
    int64_t index = -1;
    uint64_t src = ggi->gen_edges[i];
    uint64_t dst = ggi->gen_edges[i+1];
#pragma omp atomic capture
  { index = temp_counts[src]; temp_counts[src]++; }
    out_edges[index] = dst;
  
#pragma omp atomic capture
  { index = temp_counts[dst]; temp_counts[dst]++; }
    out_edges[index] = src;
  }

  free(ggi->gen_edges);
  free(temp_counts);
  g->out_edges = out_edges;
  g->out_offsets = out_offsets;

  g->local_unmap = (uint64_t*)malloc(g->n_local*sizeof(uint64_t));  
  if (g->local_unmap == NULL)
    color_throw_err("create_graph_serial(), unable to allocate unmap\n", procid);

#pragma omp parallel for
  for (uint64_t i = 0; i < g->n_local; ++i)
    g->local_unmap[i] = i + g->n_offset;

  init_map_nohash(g->map, g->n);
  set_map_id(g->map);

  if (debug) {
    printf("Task %d create_graph_serial() done: %lf (s)\n", 
      procid, omp_get_wtime() - elt);
  }
  
  return 0;
}


int clear_graph(color_dist_graph_t *g)
{
  if (debug) { printf("Task %d clear_graph() start\n", procid); }

  free(g->out_edges);
  free(g->out_offsets);
  free(g->local_unmap);
  clear_map(g->map);
  if (nprocs > 1) {
    free(g->ghost_unmap);
    free(g->ghost_tasks);
  }

  if (debug) { printf("Task %d clear_graph() success\n", procid); }
  return 0;
} 


int relabel_edges(color_dist_graph_t *g)
{
  double elt = 0.0;
  if (debug) { 
    printf("Task %d relabel_edges() start\n", procid);
    MPI_Barrier(MPI_COMM_WORLD);
    elt = omp_get_wtime();
  }

  uint64_t init_size = g->m_local < g->n ? g->m_local : g->n;
  if (init_size == g->n) {
    if (debug) printf("Task %d map init_size: %lu\n", procid, g->n);
    init_map_nohash(g->map, g->n);
  } else {
    init_size = (uint64_t)((double)init_size * 1.5);
    if (debug) printf("Task %d map init_size: %lu\n", procid, init_size);
    init_map(g->map, init_size);
  }

  for (uint64_t i = 0; i < g->n_local; ++i) {
    uint64_t vert = g->local_unmap[i];
    uint64_t local_id = i;
    set_value(g->map, vert, local_id);
  }

  uint64_t cur_label = g->n_local;
  for (uint64_t i = 0; i < g->m_local; ++i) {
    uint64_t out = g->out_edges[i];
    uint64_t local_id = get_value(g->map, out);
    if (local_id == NULL_KEY) {
      set_value(g->map, out, cur_label);
      ++cur_label;
    }
  }
  g->n_ghost = cur_label - g->n_local;
  g->n_total = g->n_ghost + g->n_local;

  if (debug) printf("Task %d, n_ghost %lu n_total %lu\n", 
    procid, g->n_ghost, g->n_total);

  g->ghost_unmap = (uint64_t*)malloc(g->n_ghost*sizeof(uint64_t));
  g->ghost_tasks = (uint64_t*)malloc(g->n_ghost*sizeof(uint64_t));
  if (g->ghost_unmap == NULL || g->ghost_tasks == NULL)
    color_throw_err("relabel_edges(), unable to allocate ghost unmaps", procid);

//#pragma omp parallel for
  for (uint64_t i = 0; i < g->n_ghost; ++i)
    g->ghost_unmap[i] = NULL_KEY;

    uint64_t n_per_rank = g->n / (uint64_t)nprocs + 1;
//#pragma omp parallel for
  for (uint64_t i = 0; i < g->m_local; ++i) {
    uint64_t global_id = g->out_edges[i];
    uint64_t local_id = get_value(g->map, global_id);
    g->out_edges[i] = local_id;
    assert(local_id < g->n_total);

    if (local_id >= g->n_local) {
      g->ghost_unmap[local_id - g->n_local] = global_id;
      g->ghost_tasks[local_id - g->n_local] = (uint64_t)(global_id / n_per_rank);
    }
  }

  if (debug) {
    printf("Task %d relabel_edges() done: %lf (s)\n", 
      procid, omp_get_wtime() - elt);
  }
  
  return 0;
}

