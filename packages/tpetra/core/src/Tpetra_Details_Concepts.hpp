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

#ifndef TPETRA_DETAILS_CONCEPTS_HPP
#define TPETRA_DETAILS_CONCEPTS_HPP

#include <type_traits>

namespace Kokkos {
namespace Compat {

template<class ExecutionSpace, class MemorySpace>
class KokkosDeviceWrapperNode;

} // namespace Compat
} // namespace Kokkos

namespace Tpetra {
namespace Details {

// Copied from KOKKOS_IMPL_IS_CONCEPT.  Define an "is_ ## CONCEPT"
// function.  See below for examples.
//
// Kokkos::Device specializations, Kokkos execution spaces, Kokkos
// memory spaces, and the old Node types all have the same three
// typedefs: device_type, execution_space, and memory_space.  Thus, we
// can't just check whether T has the desired typedef.  The definition
// of "have" resolves this, by testing whether the execution_space
// typedef has the same type as T.
#define TPETRA_DETAILS_IS_CONCEPT( CONCEPT ) \
  template< typename T > struct is_ ## CONCEPT { \
  private: \
    template< typename , typename = std::true_type > struct have : std::false_type {}; \
    template< typename U > struct have<U,typename std::is_same<U,typename U:: CONCEPT >::type> : std::true_type {}; \
  public: \
    enum { value = is_ ## CONCEPT::template have<T>::value }; \
  };

// Define "is_node" concept.  is_node<T>::value is true if and only if
// T is a WrapperNode<T> for some T.  Literally, a class is a node if
// and only if it has a typedef "node" that is the same as itself.
TPETRA_DETAILS_IS_CONCEPT( node )

// Define "is_execution_space" concept (Kokkos already has this).
// Literally, a class is an execution space if and only if it has a
// typedef "execution_space" that is the same as itself.
TPETRA_DETAILS_IS_CONCEPT( execution_space )

// Define "is_memory_space" concept (Kokkos already has this).
// Literally, a class is a memory space if and only if it has a
// typedef "memory_space" that is the same as itself.
TPETRA_DETAILS_IS_CONCEPT( memory_space )

// Define "is_device_type" concept.  Literally, a class is a device if
// and only if it has a typedef "device_type" that is the same as
// itself.
TPETRA_DETAILS_IS_CONCEPT( device_type )

// Does T "act like" a Kokkos::Device specialization, in terms of typedefs?
template<class T>
struct is_device_like {
  enum { value = is_node<T>::value ||
         is_execution_space<T>::value ||
         is_memory_space<T>::value ||
         is_device_type<T>::value };
};

// Map from NT to the corresponding KokkosDeviceWrapperNode type.  NT
// must be a KokkosDeviceWrapperNode, a Kokkos::Device, or a Kokkos
// execution or memory space.
template<class NT, bool device_like = is_device_like<NT>::value>
struct NodeType;

template<class NT>
struct NodeType<NT, true> {
  using type = ::Kokkos::Compat::KokkosDeviceWrapperNode<
    typename NT::execution_space,
    typename NT::memory_space>;
};

template<class NT>
struct NodeType<NT, false> {
  using type = NT;
};

// Whether Scalar is a valid Scalar type, for Tpetra classes that take
// a Scalar template parameter.
template<class Scalar>
struct is_valid_scalar {
  // We can't check ! std::is_integral, because Tpetra unfortunately
  // allows integral Scalar types for Various Reasons.
  enum { value = ! is_device_like<Scalar>::value };
};

} // namespace Details
} // namespace Tpetra

#endif // TPETRA_DETAILS_CONCEPTS_HPP
