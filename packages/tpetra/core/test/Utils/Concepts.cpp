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
// Questions? Contact Michael A. Heroux (maherou@sandia.gov)
//
// ************************************************************************
// @HEADER
*/

#include "Tpetra_TestingUtilities.hpp"
#include "Tpetra_Details_Concepts.hpp"
#include "Tpetra_Details_demangle.hpp"
#include "Tpetra_Details_DefaultTypes.hpp"
#include <complex>
#include <typeinfo>

namespace { // (anonymous)

  //
  // UNIT TESTS
  //

  TEUCHOS_UNIT_TEST( TpetraUtils, Concepts )
  {
    using Kokkos::Compat::KokkosDeviceWrapperNode;
    using Tpetra::Details::demangle;
    using std::endl;

    using default_scalar_type = Tpetra::Details::DefaultTypes::scalar_type;
    using default_local_ordinal_type = Tpetra::Details::DefaultTypes::local_ordinal_type;
    using default_global_ordinal_type = Tpetra::Details::DefaultTypes::global_ordinal_type;
    using default_node_type = Tpetra::Details::DefaultTypes::node_type;
    using default_execution_space = Tpetra::Details::DefaultTypes::execution_space;

    out << "default_scalar_type => "
        << demangle (typeid (default_scalar_type).name ()) << endl
        << "default_local_ordinal_type => "
        << demangle (typeid (default_local_ordinal_type).name ()) << endl
        << "default_global_ordinal_type => "
        << demangle (typeid (default_global_ordinal_type).name ()) << endl
        << "default_execution_space => "
        << demangle (typeid (default_execution_space).name ()) << endl
        << "default_node_type => "
        << demangle (typeid (default_node_type).name ()) << endl
        << endl;

    // Test is_node concept.
    using Tpetra::Details::is_node;
    static_assert (is_node<default_node_type>::value, "is_node");
    static_assert (! is_node<default_execution_space>::value, "is_node");
    static_assert (! is_node<double>::value, "is_node");
    static_assert (! is_node<int>::value, "is_node");
    static_assert (! is_node<Kokkos::HostSpace>::value, "is_node");
#ifdef KOKKOS_ENABLE_SERIAL
    static_assert (! is_node<Kokkos::Serial>::value, "is_node");
    static_assert (is_node<KokkosDeviceWrapperNode<Kokkos::Serial> >::value, "is_node");
#endif // KOKKOS_ENABLE_SERIAL
#ifdef KOKKOS_ENABLE_OPENMP
    static_assert (! is_node<Kokkos::OpenMP>::value, "is_node");
    static_assert (is_node<KokkosDeviceWrapperNode<Kokkos::OpenMP> >::value, "is_node");
#endif // KOKKOS_ENABLE_OPENMP
#ifdef KOKKOS_ENABLE_CUDA
    static_assert (! is_node<Kokkos::Cuda>::value, "is_node");
    static_assert (is_node<KokkosDeviceWrapperNode<Kokkos::Cuda> >::value, "is_node");
    static_assert (! is_node<Kokkos::CudaSpace>::value, "is_node");
    static_assert (! is_node<Kokkos::CudaUVMSpace>::value, "is_node");
#endif // KOKKOS_ENABLE_CUDA

    // Test is_execution_space concept.
    using Tpetra::Details::is_execution_space;
    static_assert (! is_execution_space<default_node_type>::value, "is_execution_space");
    static_assert (is_execution_space<default_execution_space>::value, "is_execution_space");
    static_assert (! is_execution_space<double>::value, "is_execution_space");
    static_assert (! is_execution_space<int>::value, "is_execution_space");
    static_assert (! is_execution_space<Kokkos::HostSpace>::value, "is_execution_space");
#ifdef KOKKOS_ENABLE_SERIAL
    static_assert (is_execution_space<Kokkos::Serial>::value, "is_execution_space");
    static_assert (! is_execution_space<KokkosDeviceWrapperNode<Kokkos::Serial> >::value, "is_execution_space");
#endif // KOKKOS_ENABLE_SERIAL
#ifdef KOKKOS_ENABLE_OPENMP
    static_assert (is_execution_space<Kokkos::OpenMP>::value, "is_execution_space");
    static_assert (! is_execution_space<KokkosDeviceWrapperNode<Kokkos::OpenMP> >::value, "is_execution_space");
#endif // KOKKOS_ENABLE_OPENMP
#ifdef KOKKOS_ENABLE_CUDA
    static_assert (is_execution_space<Kokkos::Cuda>::value, "is_execution_space");
    static_assert (! is_execution_space<KokkosDeviceWrapperNode<Kokkos::Cuda> >::value, "is_execution_space");
    static_assert (! is_execution_space<Kokkos::CudaSpace>::value, "is_execution_space");
    static_assert (! is_execution_space<Kokkos::CudaUVMSpace>::value, "is_execution_space");
#endif // KOKKOS_ENABLE_CUDA

    // Test is_memory_space concept.
    using Tpetra::Details::is_memory_space;
    static_assert (! is_memory_space<default_node_type>::value, "is_memory_space");
    static_assert (! is_memory_space<default_execution_space>::value, "is_memory_space");
    static_assert (! is_memory_space<double>::value, "is_memory_space");
    static_assert (! is_memory_space<int>::value, "is_memory_space");
    static_assert (is_memory_space<Kokkos::HostSpace>::value, "is_memory_space");
#ifdef KOKKOS_ENABLE_SERIAL
    static_assert (! is_memory_space<Kokkos::Serial>::value, "is_memory_space");
    static_assert (! is_memory_space<KokkosDeviceWrapperNode<Kokkos::Serial> >::value, "is_memory_space");
#endif // KOKKOS_ENABLE_SERIAL
#ifdef KOKKOS_ENABLE_OPENMP
    static_assert (! is_memory_space<Kokkos::OpenMP>::value, "is_memory_space");
    static_assert (! is_memory_space<KokkosDeviceWrapperNode<Kokkos::OpenMP> >::value, "is_memory_space");
#endif // KOKKOS_ENABLE_OPENMP
#ifdef KOKKOS_ENABLE_CUDA
    static_assert (! is_memory_space<Kokkos::Cuda>::value, "is_memory_space");
    static_assert (! is_memory_space<KokkosDeviceWrapperNode<Kokkos::Cuda> >::value, "is_memory_space");
    static_assert (is_memory_space<Kokkos::CudaSpace>::value, "is_memory_space");
    static_assert (is_memory_space<Kokkos::CudaUVMSpace>::value, "is_memory_space");
#endif // KOKKOS_ENABLE_CUDA

    // Test is_device_like concept.
    using Tpetra::Details::is_device_like;
    static_assert (is_device_like<default_node_type>::value, "is_device_like");
    static_assert (is_device_like<default_execution_space>::value, "is_device_like");
    static_assert (! is_device_like<double>::value, "is_device_like");
    static_assert (! is_device_like<int>::value, "is_device_like");
    static_assert (is_device_like<Kokkos::HostSpace>::value, "is_device_like");
#ifdef KOKKOS_ENABLE_SERIAL
    static_assert (is_device_like<Kokkos::Serial>::value, "is_device_like");
    static_assert (is_device_like<KokkosDeviceWrapperNode<Kokkos::Serial> >::value, "is_device_like");
#endif // KOKKOS_ENABLE_SERIAL
#ifdef KOKKOS_ENABLE_OPENMP
    static_assert (is_device_like<Kokkos::OpenMP>::value, "is_device_like");
    static_assert (is_device_like<KokkosDeviceWrapperNode<Kokkos::OpenMP> >::value, "is_device_like");
#endif // KOKKOS_ENABLE_OPENMP
#ifdef KOKKOS_ENABLE_CUDA
    static_assert (is_device_like<Kokkos::Cuda>::value, "is_device_like");
    static_assert (is_device_like<KokkosDeviceWrapperNode<Kokkos::Cuda> >::value, "is_device_like");
    static_assert (is_device_like<Kokkos::CudaSpace>::value, "is_device_like");
    static_assert (is_device_like<Kokkos::CudaUVMSpace>::value, "is_device_like");
#endif // KOKKOS_ENABLE_CUDA

    // Test is_valid_scalar concept.
    using Tpetra::Details::is_valid_scalar;
    static_assert (is_valid_scalar<double>::value, "is_valid_scalar");
    static_assert (is_valid_scalar<float>::value, "is_valid_scalar");
    static_assert (is_valid_scalar<std::complex<double> >::value, "is_valid_scalar");
    static_assert (is_valid_scalar<std::complex<float> >::value, "is_valid_scalar");
    // Unfortunately, Tpetra needs to support integer Scalar types.
    static_assert (is_valid_scalar<short>::value, "is_valid_scalar");
    static_assert (is_valid_scalar<unsigned short>::value, "is_valid_scalar");
    static_assert (is_valid_scalar<int>::value, "is_valid_scalar");
    static_assert (is_valid_scalar<unsigned int>::value, "is_valid_scalar");
    static_assert (is_valid_scalar<long>::value, "is_valid_scalar");
    static_assert (is_valid_scalar<unsigned long>::value, "is_valid_scalar");
    static_assert (is_valid_scalar<long long>::value, "is_valid_scalar");
    static_assert (is_valid_scalar<unsigned long long>::value, "is_valid_scalar");

    static_assert (! is_valid_scalar<Kokkos::HostSpace>::value, "is_valid_scalar");
#ifdef KOKKOS_ENABLE_SERIAL
    static_assert (! is_valid_scalar<Kokkos::Serial>::value, "is_valid_scalar");
#endif // KOKKOS_ENABLE_SERIAL
#ifdef KOKKOS_ENABLE_OPENMP
    static_assert (! is_valid_scalar<Kokkos::OpenMP>::value, "is_valid_scalar");
#endif // KOKKOS_ENABLE_OPENMP
#ifdef KOKKOS_ENABLE_CUDA
    static_assert (! is_valid_scalar<Kokkos::Cuda>::value, "is_valid_scalar");
    static_assert (! is_valid_scalar<Kokkos::CudaSpace>::value, "is_valid_scalar");
    static_assert (! is_valid_scalar<Kokkos::CudaUVMSpace>::value, "is_valid_scalar");
#endif // KOKKOS_ENABLE_CUDA
  }

} // namespace (anonymous)


