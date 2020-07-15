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
#include <algorithm>

#include "io_pp.h"
#include "dist_graph.h"
#include "util.h"

extern int procid, nprocs;
extern bool verbose, debug;


int load_graph_edges(const char *input_filename, graph_gen_data_t *ggi) 
{
  double elt = 0.0;
  if (debug) {
    printf("Task %d load_graph_edges() start\n", procid);
    MPI_Barrier(MPI_COMM_WORLD);
    elt = omp_get_wtime();
  }

  if (debug) {
    printf("Task %d reading: %s\n", procid, input_filename);
  }

  FILE* infp = fopen(input_filename, "r");
  if(infp == NULL)
    color_throw_err("load_graph_edges() unable to open input file", procid);

  fseek(infp, 0L, SEEK_END);
  uint64_t file_size = ftell(infp);
  fseek(infp, 0L, SEEK_SET);

  uint64_t nedges_global = file_size / (2*sizeof(uint64_t));
  ggi->m = nedges_global;

  uint64_t read_offset_start = 
            (uint64_t)procid*2*sizeof(uint64_t)*(nedges_global/nprocs);
  uint64_t read_offset_end = 
            (uint64_t)(procid+1)*2*sizeof(uint64_t)*(nedges_global/nprocs);
  if (procid == nprocs - 1)
    read_offset_end = 2*sizeof(uint64_t)*nedges_global;

  uint64_t nedges = (read_offset_end - read_offset_start)/(2*sizeof(uint64_t));
  ggi->m_local_read = nedges;

  if (debug) {
    printf("Task %d, read_offset_start %ld, read_offset_end %ld, nedges_global %ld, nedges: %ld\n", 
      procid, read_offset_start, read_offset_end, nedges_global, nedges);
  }

  uint64_t* gen_edges = (uint64_t*)malloc(2*nedges*sizeof(uint64_t));
  if (gen_edges == NULL)
    color_throw_err("load_graph_edges(), unable to allocate buffer", procid);

  fseek(infp, read_offset_start, SEEK_SET);
  fread(gen_edges, nedges, 2*sizeof(uint64_t), infp);
  fclose(infp);

  ggi->gen_edges = gen_edges;

  if (debug) {
    printf("Task %d read %ld edges, %9.6f (s)\n", 
      procid, nedges, omp_get_wtime() - elt);
  }
  
  uint64_t max_n = 0;
  for (uint64_t i = 0; i < ggi->m_local_read*2; ++i)
    if (gen_edges[i] > max_n)
      max_n = gen_edges[i];

  MPI_Allreduce(MPI_IN_PLACE, &max_n, 1, MPI_UINT64_T, MPI_MAX, MPI_COMM_WORLD);
  
  ggi->n = max_n+1;
  ggi->n_offset = procid*(ggi->n/nprocs );//+ 1);
  ggi->n_local = ggi->n/nprocs ;//+ 1;
  if (procid == nprocs - 1)
    ggi->n_local = ggi->n - ggi->n_offset; 
  
  if (debug) {
    for(uint64_t i = 0; i < ggi->m_local_read*2; i+=2){
      printf("Task %d: gen_edges[%lu] = %lu, gen_edges[%lu] = %lu\n",procid,i,gen_edges[i],i+1,gen_edges[i+1]);
    }
    printf("Task %d, n %lu, n_offset %lu, n_local %lu, nprocs %d\n", 
      procid, ggi->n, ggi->n_offset, ggi->n_local, nprocs);
    printf("Task %d load_graph_edges() done: %lf (s)\n", 
      procid, omp_get_wtime() - elt);
  }

  return 0;
}


int load_graph_edges_threaded(const char *input_filename, graph_gen_data_t *ggi) 
{
  double elt = 0.0;
  if (debug) {
    printf("Task %d load_graph_edges_threaded() start\n", procid);
    MPI_Barrier(MPI_COMM_WORLD);
    elt = omp_get_wtime();
  }

  if (debug) {
    printf("Task %d reading: %s\n", procid, input_filename);
  }

  FILE* infp = fopen(input_filename, "r");
  if(infp == NULL)
    color_throw_err("load_graph_edges_threaded() unable to open input file", procid);

  fseek(infp, 0L, SEEK_END);
  uint64_t file_size = ftell(infp);
  fclose(infp);

  uint64_t nedges = file_size/(2*sizeof(uint64_t));

  ggi->m = nedges;
  ggi->m_local_read = nedges;

  if (debug) {
    printf("Task %d, nedges %ld\n", procid, nedges);
  }

  uint64_t* gen_edges = (uint64_t*)malloc(2*nedges*sizeof(uint64_t));
  if (gen_edges == NULL)
    color_throw_err("load_graph_edges_threaded(), unable to allocate buffer", procid);

  uint64_t read_edges = 0;
#pragma omp parallel reduction(+:read_edges)
{
  uint64_t nthreads = (uint64_t)omp_get_num_threads();
  uint64_t tid = (uint64_t)omp_get_thread_num();

  uint64_t t_offset = tid*2*sizeof(uint64_t)*(nedges / nthreads);
  uint64_t t_offset_end = (tid+1)*2*sizeof(uint64_t)*(nedges / nthreads);
  if (tid == nthreads - 1)
    t_offset_end = 2*sizeof(uint64_t)*nedges;
  uint64_t t_nedges = (t_offset_end - t_offset)/(2*sizeof(uint64_t));
  uint64_t gen_offset = t_offset / (sizeof(uint64_t));
  uint64_t* t_gen_edges = gen_edges + gen_offset;

  FILE* t_infp = fopen(input_filename, "r");
  fseek(t_infp, t_offset, SEEK_SET);
  read_edges = fread(t_gen_edges, 2*sizeof(uint64_t), t_nedges, t_infp);
  fclose(t_infp);
} // end parallel

  ggi->gen_edges = gen_edges;

  if (debug) {
    printf("Task %d read %lu / %lu edges, %9.6f (s)\n", procid, 
      read_edges, nedges, omp_get_wtime() - elt);
  }
  
  uint64_t max_n = 0;
#pragma omp parallel for reduction(max:max_n)
  for (uint64_t i = 0; i < nedges*2; ++i) {
    if (gen_edges[i] > max_n)
      max_n = gen_edges[i];
  }
  
  ggi->n = max_n+1;
  ggi->n_offset = 0;
  ggi->n_local = ggi->n;

  if (debug) {
    printf("Task %d, n %lu, n_offset %lu, n_local %lu\n", 
      procid, ggi->n, ggi->n_offset, ggi->n_local);
    printf("Task %d load_graph_edges_threaded() done: %lf (s)\n", 
      procid, omp_get_wtime() - elt);
  }

  return 0;
}


int load_graph_edges_split(char *input_prefix, graph_gen_data_t *ggi) 
{
  double elt = 0.0;
  if (debug) {
    printf("Task %d load_graph_edges_split() start\n", procid);
    MPI_Barrier(MPI_COMM_WORLD);
    elt = omp_get_wtime();
  }

  char temp[1024]; sprintf(temp, ".%d", procid);
  char input_filename[1024]; input_filename[0] = '\0';
  strcat(input_filename, input_prefix);
  strcat(input_filename, temp);

  if (debug) {
    printf("Task %d reading: %s\n", procid, input_filename);
  }

  FILE* infp = fopen(input_filename, "r");
  if(infp == NULL)
    color_throw_err("load_graph_edges_split() unable to open input file", procid);

  fseek(infp, 0L, SEEK_END);
  uint64_t file_size = ftell(infp);

  uint64_t nedges = file_size/(2*sizeof(uint64_t));
  uint64_t nedges_global = 0;

  MPI_Allreduce(&nedges, &nedges_global, 1, 
    MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);

  ggi->m = nedges_global;
  ggi->m_local_read = nedges;

  if (debug) {
    printf("Task %d, nedges_global %ld, nedges: %ld\n", 
      procid, nedges_global, nedges);
  }

  uint64_t* gen_edges = (uint64_t*)malloc(2*nedges*sizeof(uint64_t));
  if (gen_edges == NULL)
    color_throw_err("load_graph_edges_split(), unable to allocate buffer", procid);

  fseek(infp, 0L, SEEK_SET);
  fread(gen_edges, nedges, 2*sizeof(uint64_t), infp);
  fclose(infp);

  ggi->gen_edges = gen_edges;

  if (debug) {
    printf("Task %d read %lu edges, %9.6f (s)\n",
      procid, nedges, omp_get_wtime() - elt);
  }
  
  uint64_t max_n = 0;
#pragma omp parallel for reduction(max:max_n)
  for (uint64_t i = 0; i < ggi->m_local_read*2; ++i)
    if (gen_edges[i] > max_n)
      max_n = gen_edges[i];

  MPI_Allreduce(MPI_IN_PLACE, &max_n, 1, MPI_UINT64_T, MPI_MAX, MPI_COMM_WORLD);
  
  ggi->n = max_n+1;
  ggi->n_offset = procid*(ggi->n/nprocs + 1);
  ggi->n_local = ggi->n/nprocs + 1;
  if (procid == nprocs - 1)
    ggi->n_local = ggi->n - ggi->n_offset; 

  if (debug) {
    printf("Task %d, n %lu, n_offset %lu, n_local %lu\n", 
           procid, ggi->n, ggi->n_offset, ggi->n_local);
    printf("Task %d load_graph_edges_split() done: %lf (s)\n", 
      procid, omp_get_wtime() - elt);
  }

  return 0;
}


/*int write_graph(dist_graph_t* g, char* out_prefix) 
{
  double elt = omp_get_wtime();
  if (debug) {
    printf("Task %d write_graph() start\n", procid);
  }

  char temp[1024]; sprintf(temp, ".%d", procid);
  char filename[1024]; filename[0] = '\0';
  strcat(filename, out_prefix);
  if (nprocs > 1) strcat(filename, temp);

  FILE* fp = fopen(filename, "w");

  if (debug) {
    printf("Task %d writing: %s\n", procid, filename);
  }

  fwrite(&g->n, sizeof(uint64_t), 1, fp);
  fwrite(&g->m, sizeof(uint64_t), 1, fp);
  fwrite(&g->n_local, sizeof(uint64_t), 1, fp);
  fwrite(&g->m_local, sizeof(uint64_t), 1, fp);

  fwrite(&g->n_offset, sizeof(uint64_t), 1, fp);
  fwrite(&g->n_ghost, sizeof(uint64_t), 1, fp);
  fwrite(&g->n_total, sizeof(uint64_t), 1, fp);

  fwrite(g->out_edges, sizeof(uint64_t), g->m_local, fp);
  fwrite(g->out_degree_list, sizeof(uint64_t), g->n_local+1, fp);

  fwrite(g->local_unmap, sizeof(uint64_t), g->n_local, fp);
  fwrite(g->ghost_unmap, sizeof(uint64_t), g->n_ghost, fp);
  fwrite(g->ghost_tasks, sizeof(int32_t), g->n_ghost, fp);

  fwrite(&g->map->capacity, sizeof(uint64_t), 1, fp);
  fwrite(&g->map->hashing, sizeof(bool), 1, fp);
  fwrite(g->map->arr, sizeof(uint64_t), g->map->capacity*2, fp);

  fclose(fp);

  if (debug) {
    printf("Task %d write_graph() done: %lf (s)\n", 
      procid, omp_get_wtime() - elt);
  }

  return 0;
}


int read_graph(dist_graph_t* g, char* in_prefix) 
{
  double elt = omp_get_wtime();
  if (debug) {
    printf("%d -- read_graph() start ... \n", procid);
  }

  char temp[1024]; sprintf(temp, ".%d", procid);
  char filename[1024]; filename[0] = '\0';
  strcat(filename, in_prefix);
  if (nprocs > 1) strcat(filename, temp);

  FILE* fp = fopen(filename, "r");

  if (debug) {
    printf("Task %d reading: %s\n", procid, filename);
  }

  fread(&g->n, sizeof(uint64_t), 1, fp);
  fread(&g->m, sizeof(uint64_t), 1, fp);
  fread(&g->n_local, sizeof(uint64_t), 1, fp);
  fread(&g->m_local, sizeof(uint64_t), 1, fp);

  fread(&g->n_offset, sizeof(uint64_t), 1, fp);
  fread(&g->n_ghost, sizeof(uint64_t), 1, fp);
  fread(&g->n_total, sizeof(uint64_t), 1, fp);

  g->out_edges = (uint64_t*)malloc(g->m_local*sizeof(uint64_t));
  g->out_degree_list = (uint64_t*)malloc((g->n_local+1)*sizeof(uint64_t));
  fread(g->out_edges, sizeof(uint64_t), g->m_local, fp);
  fread(g->out_degree_list, sizeof(uint64_t), g->n_local+1, fp);

  g->local_unmap = (uint64_t*)malloc(g->n_local*sizeof(uint64_t));
  g->ghost_unmap = (uint64_t*)malloc(g->n_ghost*sizeof(uint64_t));
  g->ghost_tasks = (uint64_t*)malloc(g->n_ghost*sizeof(uint64_t));
  fread(g->local_unmap, sizeof(uint64_t), g->n_local, fp);
  fread(g->ghost_unmap, sizeof(uint64_t), g->n_ghost, fp);
  fread(g->ghost_tasks, sizeof(uint64_t), g->n_ghost, fp);

  g->map = (fast_map*)malloc(sizeof(fast_map));
  fread(&g->map->capacity, sizeof(uint64_t), 1, fp);
  fread(&g->map->hashing, sizeof(bool), 1, fp);
  g->map->arr = (uint64_t*)malloc(g->map->capacity*2*sizeof(uint64_t));
  fread(g->map->arr, sizeof(uint64_t), g->map->capacity*2, fp);

  fclose(fp);

  if (debug) {
    printf("%d -- read_graph() done: %lf (s)\n", 
      procid, omp_get_wtime() - elt);
  }

  return 0;
}*/


int exchange_edges(graph_gen_data_t *ggi)
{
  double elt = 0.0;
  if (debug) {
    printf("Task %d exchange_edges() start\n", procid);
    MPI_Barrier(MPI_COMM_WORLD);
    elt = omp_get_wtime();
  }

  int* sendcounts = (int*)malloc(nprocs*sizeof(int));
  int* recvcounts = (int*)malloc(nprocs*sizeof(int));
  int* sdispls = (int*)malloc((nprocs+1)*sizeof(int));
  int* sdispls_cpy = (int*)malloc((nprocs+1)*sizeof(int));
  int* rdispls = (int*)malloc((nprocs+1)*sizeof(int));

  uint64_t* temp_sendcounts = (uint64_t*)malloc(nprocs*sizeof(uint64_t));
  uint64_t* temp_recvcounts = (uint64_t*)malloc(nprocs*sizeof(uint64_t));
  for (int i = 0; i < nprocs; ++i) {
    temp_sendcounts[i] = 0;
    temp_recvcounts[i] = 0;
  }

  uint64_t n_per_rank = ggi->n / nprocs ;//+ 1;
  uint64_t* thread_sendcounts = 
    (uint64_t*)malloc(omp_get_max_threads()*(uint64_t)nprocs*sizeof(uint64_t));
//#pragma omp parallel
{
  uint64_t* thread_sendcount = &thread_sendcounts[omp_get_thread_num()*nprocs];
  for (int i = 0; i < nprocs; ++i)
    thread_sendcount[i] = 0;

//#pragma omp for
  for (uint64_t i = 0; i < ggi->m_local_read*2; i+=2) {
    uint64_t vert1 = ggi->gen_edges[i];
    int32_t vert_task1 = std::min((int32_t)(vert1 / n_per_rank),nprocs-1);
    thread_sendcount[vert_task1] += 2;

    uint64_t vert2 = ggi->gen_edges[i+1];
    int32_t vert_task2 = std::min((int32_t)(vert2 / n_per_rank),nprocs-1);
    thread_sendcount[vert_task2] += 2;
  }

  for (int i = 0; i < nprocs; ++i)
//#pragma omp atomic
    temp_sendcounts[i] += thread_sendcount[i];
} // end parallel

  MPI_Alltoall(temp_sendcounts, 1, MPI_UINT64_T, 
               temp_recvcounts, 1, MPI_UINT64_T, MPI_COMM_WORLD);
  
  uint64_t total_recv = 0;
  uint64_t total_send = 0;
  for (int32_t i = 0; i < nprocs; ++i) {
    total_recv += temp_recvcounts[i];
    total_send += temp_sendcounts[i];
  }
  free(temp_sendcounts);
  free(temp_recvcounts);

  uint64_t* recvbuf = (uint64_t*)malloc(total_recv*sizeof(uint64_t));
  if (recvbuf == NULL)
  { 
    fprintf(stderr, "Task %d Error: exchange_out_edges(), unable to allocate recv buffer %lu bytes\n", procid, total_recv);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }  

  uint64_t max_transfer = total_send > total_recv ? total_send : total_recv;
  uint64_t num_comms = max_transfer / (uint64_t)(MAX_SEND_SIZE) + 1;
  MPI_Allreduce(MPI_IN_PLACE, &num_comms, 1, 
                MPI_UINT64_T, MPI_MAX, MPI_COMM_WORLD);

  if (debug) 
    printf("Task %d exchange_edges() num_comms %lu total_send %lu total_recv %lu\n", procid, num_comms, total_send, total_recv);

  uint64_t max_send = total_send / num_comms;
  max_send = (uint64_t)((double)max_send * 1.1);
  uint64_t* sendbuf = (uint64_t*)malloc(max_send*sizeof(uint64_t));
  if (sendbuf == NULL) { 
    fprintf(stderr, "Task %d Error: exchange_out_edges(), unable to allocate send buffer", procid);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  uint64_t sum_recv = 0;
  for (uint64_t c = 0; c < num_comms; ++c) {
    uint64_t send_begin = (ggi->m_local_read * c) / num_comms;
    uint64_t send_end = (ggi->m_local_read * (c + 1)) / num_comms;
    if (c == (num_comms-1))
      send_end = ggi->m_local_read;
    assert(send_end - send_begin < max_send);

    for (int32_t i = 0; i < nprocs; ++i) {
      sendcounts[i] = 0;
      recvcounts[i] = 0;
    }

//#pragma omp parallel
{
    uint64_t* thread_sendcount = 
        &thread_sendcounts[omp_get_thread_num()*nprocs];
    for (int i = 0; i < nprocs; ++i)
      thread_sendcount[i] = 0;

//#pragma omp for
    for (uint64_t i = send_begin; i < send_end; ++i) {
      uint64_t vert1 = ggi->gen_edges[i*2];
      int32_t vert_task1 = std::min((int32_t)(vert1 / n_per_rank),nprocs-1);
      thread_sendcount[vert_task1] += 2;

      uint64_t vert2 = ggi->gen_edges[i*2+1];
      int32_t vert_task2 = std::min((int32_t)(vert2 / n_per_rank),nprocs-1);
      thread_sendcount[vert_task2] += 2;

      assert(vert1 != 0 || vert2 != 0);
    }

    for (int i = 0; i < nprocs; ++i)
//#pragma omp atomic
      sendcounts[i] += thread_sendcount[i];
} // end parallel

    MPI_Alltoall(sendcounts, 1, MPI_INT32_T, 
                 recvcounts, 1, MPI_INT32_T, MPI_COMM_WORLD);
    /*for(int i = 0; i < nprocs; i++){
      printf("Task %d, sendcount[%d] = %d recvcount[%d] = %d\n",procid,i,sendcounts[i],i,recvcounts[i]);
    }*/
    
    sdispls[0] = 0;
    sdispls_cpy[0] = 0;
    rdispls[0] = 0;
    for (int32_t i = 1; i < nprocs; ++i) {
      sdispls[i] = sdispls[i-1] + sendcounts[i-1];
      rdispls[i] = rdispls[i-1] + recvcounts[i-1];
      sdispls_cpy[i] = sdispls[i];
      //printf("Task %d sdispls[%d] = %d, sendcounts[%d] = %d, recvcounts[%d] = %d\n",procid,i-1,sdispls[i-1],i-1,sendcounts[i-1], i-1,recvcounts[i-1]);
    }

    int32_t cur_send = sdispls[nprocs-1] + sendcounts[nprocs-1];
    int32_t cur_recv = rdispls[nprocs-1] + recvcounts[nprocs-1];
    assert((uint64_t)cur_send < max_send);
    if (debug) 
      printf("Task %d cur_send %d, cur_recv %d\n", procid, cur_send, cur_recv);

//#pragma omp parallel for    
    for (uint64_t i = send_begin; i < send_end; ++i) {
      uint64_t vert1 = ggi->gen_edges[2*i];
      uint64_t vert2 = ggi->gen_edges[2*i+1];
      int32_t vert_task1 = std::min((int32_t)(vert1 / n_per_rank),nprocs-1);
      int32_t vert_task2 = std::min((int32_t)(vert2 / n_per_rank),nprocs-1);

      uint64_t offset1;
      uint64_t offset2;

//#pragma omp atomic capture
    { offset1 = sdispls_cpy[vert_task1]; 
      sdispls_cpy[vert_task1] += 2; }
//#pragma omp atomic capture
    { offset2 = sdispls_cpy[vert_task2]; 
      sdispls_cpy[vert_task2] += 2; }

      sendbuf[offset1] = vert1; 
      sendbuf[offset1+1] = vert2;
      sendbuf[offset2] = vert2; 
      sendbuf[offset2+1] = vert1;
    }

    MPI_Alltoallv(sendbuf, sendcounts, sdispls, MPI_UINT64_T, 
                  recvbuf+sum_recv, recvcounts, rdispls,
                  MPI_UINT64_T, MPI_COMM_WORLD);
    sum_recv += cur_recv;
  }
  free(sendbuf);
  free(thread_sendcounts);
  free(ggi->gen_edges);
  free(sendcounts);
  free(recvcounts);
  free(sdispls);
  free(sdispls_cpy);
  free(rdispls);
  
  ggi->gen_edges = recvbuf;
  ggi->m_local_edges = total_recv / 2;

  if (debug) {   
    for(uint64_t i = 0; i < ggi->m_local_edges*2; i +=2){
      printf("Task %d: gen_edges[%lu] = %lu, gen_edges[%lu] = %lu\n",procid,i,ggi->gen_edges[i],i+1,ggi->gen_edges[i+1]);
    } 
    printf("Task %d exchange_out_edges() sent %lu, recv %lu, m_local_edges %lu, %9.6f (s)\n", 
      procid, total_send, total_recv, ggi->m_local_edges, elt);
    printf("Task %d exchange_out_edges() done: %lf (s)\n", 
      procid, omp_get_wtime() - elt);
  }

  return 0;
}
