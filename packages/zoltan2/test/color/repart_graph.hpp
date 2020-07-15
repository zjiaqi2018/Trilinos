#ifndef __REPART_DIST_GRAPH_HPP__
#define __REPART_DIST_GRAPH_HPP__

extern int procid, nprocs;

#include "dist_graph.h"
#include "util.h"
#define MAX_SEND_SIZE 2147483648
struct mpi_data_t {
  int32_t* sendcounts;
  uint64_t* sendcounts_temp;
  int32_t* recvcounts;
  uint64_t* recvcounts_temp;
  int32_t* sdispls;
  int32_t* rdispls;
  int32_t* sdispls_cpy;
  uint64_t* sdispls_temp;

  uint64_t* sendbuf_vert;
  int32_t* sendbuf_data;
  uint64_t* recvbuf_vert;
  int32_t* recvbuf_data;

  uint64_t total_recv;
  uint64_t total_send;
  uint64_t global_queue_size;
};

void color_init_comm_data(mpi_data_t* comm){
  comm->sendcounts = (int32_t*)malloc(nprocs*sizeof(int32_t));
  comm->sendcounts_temp = (uint64_t*)malloc(nprocs*sizeof(uint64_t));
  comm->recvcounts = (int32_t*)malloc(nprocs*sizeof(int32_t));
  comm->recvcounts_temp = (uint64_t*)malloc(nprocs*sizeof(uint64_t));
  comm->sdispls = (int32_t*)malloc(nprocs*sizeof(int32_t));
  comm->rdispls = (int32_t*)malloc(nprocs*sizeof(int32_t));
  comm->sdispls_cpy = (int32_t*)malloc(nprocs*sizeof(int32_t));
  comm->sdispls_temp = (uint64_t*)malloc(nprocs*sizeof(int64_t));

  if (comm->sendcounts == NULL || comm->sendcounts_temp == NULL ||
      comm->recvcounts == NULL || comm->sdispls == NULL ||
      comm->rdispls == NULL || comm->sdispls_cpy == NULL)
    color_throw_err("init_comm_data(), unable to allocate resources\n", procid);

  comm->total_recv = 0;
  comm->total_send = 0;
  comm->global_queue_size = 0;

}

void color_clear_comm_data(mpi_data_t* comm){
  free(comm->sendcounts);
  free(comm->sendcounts_temp);
  free(comm->recvcounts);
  free(comm->recvcounts_temp);
  free(comm->sdispls);
  free(comm->rdispls);
  free(comm->sdispls_cpy);
  free(comm->sdispls_temp);
  
}

void repart_graph(color_dist_graph_t *g, mpi_data_t* comm, int32_t* local_parts)
{
   for (int i = 0; i < nprocs; ++i)
  {
    comm->sendcounts_temp[i] = 0;
    comm->recvcounts_temp[i] = 0;
  }

  for (uint64_t i = 0; i < g->n_local; ++i)
  {
    int32_t rank = local_parts[i];
    ++comm->sendcounts_temp[rank];
  }

  MPI_Alltoall(comm->sendcounts_temp, 1, MPI_UINT64_T, 
               comm->recvcounts_temp, 1, MPI_UINT64_T, MPI_COMM_WORLD);
  
  uint64_t total_recv = 0;
  uint64_t total_send = 0;
  for (int32_t i = 0; i < nprocs; ++i)
  {
    total_recv += comm->recvcounts_temp[i];
    total_send += comm->sendcounts_temp[i];
  }

  uint64_t* recvbuf_vids = (uint64_t*)malloc(total_recv*sizeof(uint64_t));
  uint64_t* recvbuf_deg_out = (uint64_t*)malloc(total_recv*sizeof(uint64_t));
  //uint64_t* recvbuf_deg_in = (uint64_t*)malloc(total_recv*sizeof(uint64_t));
  if (recvbuf_vids == NULL ||
      recvbuf_deg_out == NULL )//|| recvbuf_deg_in == NULL)
    color_throw_err("repart_graph(), unable to allocate buffers\n", procid);

  uint64_t max_transfer = total_send > total_recv ? total_send : total_recv;
  uint64_t num_comms = max_transfer / (uint64_t)(MAX_SEND_SIZE/(g->m/g->n))+ 1;
  MPI_Allreduce(MPI_IN_PLACE, &num_comms, 1, 
                MPI_UINT64_T, MPI_MAX, MPI_COMM_WORLD);

  if (debug) 
    printf("Task %d repart_graph() num_comms %lu total_send %lu total_recv %lu\n", procid, num_comms, total_send, total_recv);

  uint64_t sum_recv_deg = 0;
  for (uint64_t c = 0; c < num_comms; ++c)
  {
    uint64_t send_begin = (g->n_local * c) / num_comms;
    uint64_t send_end = (g->n_local * (c + 1)) / num_comms;
    if (c == (num_comms-1))
      send_end = g->n_local;

    for (int32_t i = 0; i < nprocs; ++i)
    {
      comm->sendcounts[i] = 0;
      comm->recvcounts[i] = 0;
    }

    if (debug)
      printf("Task %d send_begin %lu send_end %lu\n", procid, send_begin, send_end);
    for (uint64_t i = send_begin; i < send_end; ++i)
    {
      int32_t rank = local_parts[i];
      ++comm->sendcounts[rank];
    }

    MPI_Alltoall(comm->sendcounts, 1, MPI_INT32_T, 
                 comm->recvcounts, 1, MPI_INT32_T, MPI_COMM_WORLD);

    comm->sdispls[0] = 0;
    comm->sdispls_cpy[0] = 0;
    comm->rdispls[0] = 0;
    for (int32_t i = 1; i < nprocs; ++i)
    {
      comm->sdispls[i] = comm->sdispls[i-1] + comm->sendcounts[i-1];
      comm->rdispls[i] = comm->rdispls[i-1] + comm->recvcounts[i-1];
      comm->sdispls_cpy[i] = comm->sdispls[i];
    }

    int32_t cur_send = comm->sdispls[nprocs-1] + comm->sendcounts[nprocs-1];
    int32_t cur_recv = comm->rdispls[nprocs-1] + comm->recvcounts[nprocs-1];
    uint64_t* sendbuf_vids = (uint64_t*)malloc((uint64_t)cur_send*sizeof(uint64_t));
    uint64_t* sendbuf_deg_out = (uint64_t*)malloc((uint64_t)cur_send*sizeof(uint64_t));
   // uint64_t* sendbuf_deg_in = (uint64_t*)malloc((uint64_t)cur_send*sizeof(uint64_t));
    if (sendbuf_vids == NULL || 
        sendbuf_deg_out == NULL )//|| sendbuf_deg_in == NULL)
      color_throw_err("repart_graph(), unable to allocate buffers\n", procid);

    for (uint64_t i = send_begin; i < send_end; ++i)
    {
      int32_t rank = local_parts[i];
      int32_t snd_index = comm->sdispls_cpy[rank]++;
      sendbuf_vids[snd_index] = g->local_unmap[i];
      sendbuf_deg_out[snd_index] = (uint64_t)color_out_degree(g, i);
      //sendbuf_deg_in[snd_index] = (uint64_t)in_degree(g, i);
    }

    MPI_Alltoallv(
      sendbuf_vids, comm->sendcounts, comm->sdispls, MPI_UINT64_T, 
      recvbuf_vids+sum_recv_deg, comm->recvcounts, comm->rdispls,
      MPI_UINT64_T, MPI_COMM_WORLD);    
    MPI_Alltoallv(
      sendbuf_deg_out, comm->sendcounts, comm->sdispls, MPI_UINT64_T, 
      recvbuf_deg_out+sum_recv_deg, comm->recvcounts, comm->rdispls,
      MPI_UINT64_T, MPI_COMM_WORLD);
    /*MPI_Alltoallv(
      sendbuf_deg_in, comm->sendcounts, comm->sdispls, MPI_UINT64_T, 
      recvbuf_deg_in+sum_recv_deg, comm->recvcounts, comm->rdispls,
      MPI_UINT64_T, MPI_COMM_WORLD);*/
    sum_recv_deg += (uint64_t)cur_recv;
    free(sendbuf_vids);
    free(sendbuf_deg_out);
    //free(sendbuf_deg_in);
  }

  for (int i = 0; i < nprocs; ++i)
  {
    comm->sendcounts_temp[i] = 0;
    comm->recvcounts_temp[i] = 0;
  }

  for (uint64_t i = 0; i < g->n_local; ++i)
  {
    int32_t rank = local_parts[i];
    comm->sendcounts_temp[rank] += (uint64_t)color_out_degree(g, i);
  }

  MPI_Alltoall(comm->sendcounts_temp, 1, MPI_UINT64_T, 
               comm->recvcounts_temp, 1, MPI_UINT64_T, MPI_COMM_WORLD);
  
  total_recv = 0;
  total_send = 0;
  for (int32_t i = 0; i < nprocs; ++i)
  {
    total_recv += comm->recvcounts_temp[i];
    total_send += comm->sendcounts_temp[i];
  }

  uint64_t* recvbuf_e_out = (uint64_t*)malloc(total_recv*sizeof(uint64_t));
  if (recvbuf_e_out == NULL)
    color_throw_err("repart_graph(), unable to allocate buffer\n", procid);

  // max_transfer = total_send > total_recv ? total_send : total_recv;
  // num_comms = max_transfer / (uint64_t)MAX_SEND_SIZE + 1;
  // MPI_Allreduce(MPI_IN_PLACE, &num_comms, 1, 
  //               MPI_UINT64_T, MPI_MAX, MPI_COMM_WORLD);

  if (debug) 
    printf("Task %d repart_graph() num_comms %lu total_send %lu total_recv %lu\n", procid, num_comms, total_send, total_recv);
  
  uint64_t sum_recv_e_out = 0;
  for (uint64_t c = 0; c < num_comms; ++c)
  {
    uint64_t send_begin = (g->n_local * c) / num_comms;
    uint64_t send_end = (g->n_local * (c + 1)) / num_comms;
    if (c == (num_comms-1))
      send_end = g->n_local;

    for (int32_t i = 0; i < nprocs; ++i)
    {
      comm->sendcounts[i] = 0;
      comm->recvcounts[i] = 0;
    }

    for (uint64_t i = send_begin; i < send_end; ++i)
    {
      uint32_t rank = local_parts[i];
      comm->sendcounts[rank] += (int32_t)color_out_degree(g, i);
    }

    MPI_Alltoall(comm->sendcounts, 1, MPI_INT32_T, 
                 comm->recvcounts, 1, MPI_INT32_T, MPI_COMM_WORLD);

    comm->sdispls[0] = 0;
    comm->sdispls_cpy[0] = 0;
    comm->rdispls[0] = 0;
    for (int32_t i = 1; i < nprocs; ++i)
    {
      comm->sdispls[i] = comm->sdispls[i-1] + comm->sendcounts[i-1];
      comm->rdispls[i] = comm->rdispls[i-1] + comm->recvcounts[i-1];
      comm->sdispls_cpy[i] = comm->sdispls[i];
    }

    int32_t cur_send = comm->sdispls[nprocs-1] + comm->sendcounts[nprocs-1];
    int32_t cur_recv = comm->rdispls[nprocs-1] + comm->recvcounts[nprocs-1];
    uint64_t* sendbuf_e_out = (uint64_t*)malloc((uint64_t)cur_send*sizeof(uint64_t));
    if (sendbuf_e_out == NULL)
      color_throw_err("repart_graph(), unable to allocate buffer\n", procid);

    for (uint64_t i = send_begin; i < send_end; ++i)
    {
      uint64_t color_out_degree = color_out_degree(g, i);
      uint64_t* outs = color_out_vertices(g, i);
      int32_t rank = local_parts[i];
      int32_t snd_index = comm->sdispls_cpy[rank];
      comm->sdispls_cpy[rank] += color_out_degree;
      for (uint64_t j = 0; j < color_out_degree; ++j)
      {
        uint64_t out;
        if (outs[j] < g->n_local)
          out = g->local_unmap[outs[j]];
        else
          out = g->ghost_unmap[outs[j]-g->n_local];
        sendbuf_e_out[snd_index++] = out;
      }
    }

    MPI_Alltoallv(sendbuf_e_out, comm->sendcounts, comm->sdispls, MPI_UINT64_T, 
                  recvbuf_e_out+sum_recv_e_out, comm->recvcounts, comm->rdispls,
                  MPI_UINT64_T, MPI_COMM_WORLD);
    sum_recv_e_out += (uint64_t)cur_recv;
    free(sendbuf_e_out);
  }

  free(g->out_edges);
  free(g->out_offsets);
  g->out_edges = recvbuf_e_out;
  g->m_local = (uint64_t)sum_recv_e_out;
  g->out_offsets = (uint64_t*)malloc((sum_recv_deg+1)*sizeof(uint64_t));
  g->out_offsets[0] = 0;
  for (uint64_t i = 0; i < sum_recv_deg; ++i)
    g->out_offsets[i+1] = g->out_offsets[i] + recvbuf_deg_out[i];
  assert(g->out_offsets[sum_recv_deg] == g->m_local);
  free(recvbuf_deg_out);

  for (int i = 0; i < nprocs; ++i)
  {
    comm->sendcounts_temp[i] = 0;
    comm->recvcounts_temp[i] = 0;
  }

  /*for (uint64_t i = 0; i < g->n_local; ++i)
  {
    int32_t rank = local_parts[i];
    comm->sendcounts_temp[rank] += (uint64_t)in_degree(g, i);
  }

  MPI_Alltoall(comm->sendcounts_temp, 1, MPI_UINT64_T, 
               comm->recvcounts_temp, 1, MPI_UINT64_T, MPI_COMM_WORLD);
  
  total_recv = 0;
  total_send = 0;
  for (int32_t i = 0; i < nprocs; ++i)
  {
    total_recv += comm->recvcounts_temp[i];
    total_send += comm->sendcounts_temp[i];
  }

  uint64_t* recvbuf_e_in = (uint64_t*)malloc(total_recv*sizeof(uint64_t));
  if (recvbuf_e_in == NULL)
    color_throw_err("repart_graph(), unable to allocate buffer\n", procid);

  // max_transfer = total_send > total_recv ? total_send : total_recv;
  // num_comms = max_transfer / (uint64_t)MAX_SEND_SIZE + 1;
  // MPI_Allreduce(MPI_IN_PLACE, &num_comms, 1, 
  //               MPI_UINT64_T, MPI_MAX, MPI_COMM_WORLD);

  if (debug) 
    printf("Task %d repart_graph() num_comms %lu total_send %lu total_recv %lu\n", procid, num_comms, total_send, total_recv);
  
  uint64_t sum_recv_e_in = 0;
  for (uint64_t c = 0; c < num_comms; ++c)
  {
    uint64_t send_begin = (g->n_local * c) / num_comms;
    uint64_t send_end = (g->n_local * (c + 1)) / num_comms;
    if (c == (num_comms-1))
      send_end = g->n_local;

    for (int32_t i = 0; i < nprocs; ++i)
    {
      comm->sendcounts[i] = 0;
      comm->recvcounts[i] = 0;
    }

    for (uint64_t i = send_begin; i < send_end; ++i)
    {
      int32_t rank = local_parts[i];
      comm->sendcounts[rank] += (int32_t)in_degree(g, i);
    }

    MPI_Alltoall(comm->sendcounts, 1, MPI_INT32_T, 
                 comm->recvcounts, 1, MPI_INT32_T, MPI_COMM_WORLD);

    comm->sdispls[0] = 0;
    comm->sdispls_cpy[0] = 0;
    comm->rdispls[0] = 0;
    for (int32_t i = 1; i < nprocs; ++i)
    {
      comm->sdispls[i] = comm->sdispls[i-1] + comm->sendcounts[i-1];
      comm->rdispls[i] = comm->rdispls[i-1] + comm->recvcounts[i-1];
      comm->sdispls_cpy[i] = comm->sdispls[i];
    }

    int32_t cur_send = comm->sdispls[nprocs-1] + comm->sendcounts[nprocs-1];
    int32_t cur_recv = comm->rdispls[nprocs-1] + comm->recvcounts[nprocs-1];
    uint64_t* sendbuf_e_in = (uint64_t*)malloc((uint64_t)cur_send*sizeof(uint64_t));
    if (sendbuf_e_in == NULL)
      color_throw_err("repart_graph(), unable to allocate buffer\n", procid);

    for (uint64_t i = send_begin; i < send_end; ++i)
    {
      uint64_t in_degree = in_degree(g, i);
      uint64_t* ins = in_vertices(g, i);
      int32_t rank = local_parts[i];
      int32_t snd_index = comm->sdispls_cpy[rank];
      comm->sdispls_cpy[rank] += in_degree;
      for (uint32_t j = 0; j < in_degree; ++j)
      {
        uint64_t in;
        if (ins[j] < g->n_local)
          in = g->local_unmap[ins[j]];
        else
          in = g->ghost_unmap[ins[j]-g->n_local];
        sendbuf_e_in[snd_index++] = in;
      }
    }

    MPI_Alltoallv(sendbuf_e_in, comm->sendcounts, comm->sdispls, MPI_UINT64_T, 
                  recvbuf_e_in+sum_recv_e_in, comm->recvcounts, comm->rdispls,
                  MPI_UINT64_T, MPI_COMM_WORLD);
    sum_recv_e_in += (uint64_t)cur_recv;
    free(sendbuf_e_in);
  }

  free(g->in_edges);
  free(g->in_degree_list);
  g->in_edges = recvbuf_e_in;
  g->m_local_in = (uint64_t)sum_recv_e_in;
  g->in_degree_list = (uint64_t*)malloc((sum_recv_deg+1)*sizeof(uint64_t));
  g->in_degree_list[0] = 0;
  for (uint64_t i = 0; i < sum_recv_deg; ++i)
    g->in_degree_list[i+1] = g->in_degree_list[i] + recvbuf_deg_in[i];
  assert(g->in_degree_list[sum_recv_deg] == g->m_local_in);
  free(recvbuf_deg_in);*/

  free(g->local_unmap);
  g->local_unmap = (uint64_t*)malloc(sum_recv_deg*sizeof(uint64_t));
  for (uint64_t i = 0; i < sum_recv_deg; ++i)
    g->local_unmap[i] = recvbuf_vids[i];
  free(recvbuf_vids);

  g->n_local = sum_recv_deg;
}

#endif
