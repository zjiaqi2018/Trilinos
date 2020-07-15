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

#ifndef _FAST_MAP_H_
#define _FAST_MAP_H_

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

extern int procid, nprocs;
extern bool verbose, debug;

#define NULL_KEY 18446744073709551615U

struct color_fast_map {
  uint64_t* arr;
  uint64_t capacity;
  bool hashing;
} ;

void init_map(color_fast_map* map, uint64_t init_size);
void init_map_nohash(color_fast_map* map, uint64_t init_size);
void set_map_id(color_fast_map* map);
void clear_map(color_fast_map* map);

inline uint64_t mult_hash(color_fast_map* map, uint64_t key);
inline void set_value(color_fast_map* map, uint64_t key, uint64_t value);
inline uint64_t get_value(color_fast_map* map, uint64_t key);


inline uint64_t mult_hash(color_fast_map* map, uint64_t key)
{
  if (map->hashing)
    return (key*2654435761 % map->capacity);
  else
    return key;
}


inline void set_value(color_fast_map* map, uint64_t key, uint64_t value)
{
  if (!map->hashing) {
    map->arr[key] = value;
    return;
  }

  uint64_t cur_index = mult_hash(map, key)*2;
  uint64_t count = 0;
  uint64_t j = 0;
  while (map->arr[cur_index] != key && map->arr[cur_index] != NULL_KEY)
  {
    ++j; 
    cur_index = (cur_index + (j * j * 2)) % (map->capacity*2);
    ++count;
    if (debug && count % 100 == 0)
      fprintf(stderr, "Warning: color_fast_map set_value(): Big Count %d -- %lu - %lu, %lu, %lu\n", procid, count, cur_index, key, value);
  }
  if (map->arr[cur_index] == NULL_KEY)
  {  
    map->arr[cur_index] = key;
  }
  map->arr[cur_index+1] = value;
}


inline uint64_t get_value(color_fast_map* map, uint64_t key)
{
  if (!map->hashing) return map->arr[key];

  uint64_t cur_index = mult_hash(map, key)*2;
  uint64_t count = 0;
  uint64_t j = 0;
  while (map->arr[cur_index] != key && map->arr[cur_index] != NULL_KEY) 
  {  
    ++j;
    cur_index = (cur_index + (j * j * 2)) % (map->capacity*2);
    ++count;
    if (debug && count % 100 == 0)
      fprintf(stderr, "Warning: color_fast_map set_value_uq(): Big Count %d -- %lu - %lu, %lu\n", procid, count, cur_index, key);
  }
  if (map->arr[cur_index] == NULL_KEY)
    return NULL_KEY;
  else
    return map->arr[cur_index+1];
}


#endif
