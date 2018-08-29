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

#ifndef TPETRA_DETAILS_ALIASES_HPP
#define TPETRA_DETAILS_ALIASES_HPP

#include "Tpetra_Details_Concepts.hpp"
#include <type_traits>

namespace Tpetra {
namespace Details {

// Recurse on the list of template parameters Args... to determine
// local_ordinal_type, global_ordinal_type, and node_type.
template<class Traits,
         class Defaults,
         class ... Args>
struct ThreeArgTraitsImpl;

// Recurse on the list of template parameters Args... to determine
// scalar_type, local_ordinal_type, global_ordinal_type, and node_type.
template<class Traits,
         class Defaults,
         class ... Args>
struct FourArgTraitsImpl;

// Base case of the recursion: use Traits (some struct with typedefs)
// to get local_ordinal_type, global_ordinal_type, and node_type.  Any
// missing typedefs (represented by Traits having void for that
// typedef) get filled in with defaults from the second template
// parameter.
template<class Traits, class Defaults>
struct ThreeArgTraitsImpl<Traits, Defaults> {
private:
  static constexpr bool have_lo =
    ! std::is_same<typename Traits::local_ordinal_type, void>::value;
  static constexpr bool have_go =
    ! std::is_same<typename Traits::global_ordinal_type, void>::value;
  static constexpr bool have_nt =
    ! std::is_same<typename Traits::node_type, void>::value;

  static_assert (! have_lo || std::is_integral<typename Traits::local_ordinal_type>::value,
                 "Please report this bug to the Tpetra developers.");
  static_assert (! have_go || std::is_integral<typename Traits::global_ordinal_type>::value,
                 "Please report this bug to the Tpetra developers.");
  static_assert (! have_nt || is_node<typename Traits::node_type>::value,
                 "Please report this bug to the Tpetra developers.");

public:
  using local_ordinal_type =
    typename std::conditional<
      have_lo,
      typename Traits::local_ordinal_type,
      typename Defaults::local_ordinal_type
    >::type;
  using global_ordinal_type =
    typename std::conditional<
      have_go,
      typename Traits::global_ordinal_type,
      typename Defaults::global_ordinal_type
    >::type;
  using node_type =
    typename std::conditional<
      have_nt,
      typename Traits::node_type,
      typename Defaults::node_type
    >::type;

  static_assert (! have_lo || std::is_integral<local_ordinal_type>::value,
                 "Please report this bug to the Tpetra developers.");
  static_assert (! have_go || std::is_integral<global_ordinal_type>::value,
                 "Please report this bug to the Tpetra developers.");
  static_assert (! have_nt || is_node<node_type>::value,
                 "Please report this bug to the Tpetra developers.");
};

// Base case of the recursion: use Traits (some struct with typedefs)
// to get scalar_type, local_ordinal_type, global_ordinal_type, and
// node_type.  Any missing typedefs (represented by Traits having void
// for that typedef) get filled in with defaults from the second
// template parameter.
template<class Traits, class Defaults>
struct FourArgTraitsImpl<Traits, Defaults> {
private:
  static constexpr bool have_sc =
    ! std::is_same<typename Traits::scalar_type, void>::value;
  static constexpr bool have_lo =
    ! std::is_same<typename Traits::local_ordinal_type, void>::value;
  static constexpr bool have_go =
    ! std::is_same<typename Traits::global_ordinal_type, void>::value;
  static constexpr bool have_nt =
    ! std::is_same<typename Traits::node_type, void>::value;

public:
  using scalar_type =
    typename std::conditional<
      have_sc,
      typename Traits::scalar_type,
      typename Defaults::scalar_type
    >::type;
  using local_ordinal_type =
    typename std::conditional<
      have_lo,
      typename Traits::local_ordinal_type,
      typename Defaults::local_ordinal_type
    >::type;
  using global_ordinal_type =
    typename std::conditional<
      have_go,
      typename Traits::global_ordinal_type,
      typename Defaults::global_ordinal_type
    >::type;
  using node_type =
    typename std::conditional<
      have_nt,
      typename Traits::node_type,
      typename Defaults::node_type
    >::type;
};

// Recurse on the template parameter list.
template<class Traits,
         class Defaults,
         class First,
         class ... Rest>
struct ThreeArgTraitsImpl<Traits, Defaults, First, Rest...> {
private:
  static_assert (sizeof... (Rest) <= 2, "Tpetra classes like Map and CrsGraph "
                 "may have no more than three template parameters.");
  static_assert (is_device_like<First>::value || std::is_integral<First>::value,
                 "For Tpetra classes like Map and CrsGraph, the first "
                 "template parameter must either be a built-in integer type, "
                 "or something that can be converted to Kokkos::Device.");

  // Recursion depends on the aux Traits template parameter.  This is
  // always a struct with three typedefs: local_ordinal_type,
  // global_ordinal_type, and node_type.  The typedefs start all as
  // void.  We then fill them in as we learn them.
  static constexpr bool have_lo =
    ! std::is_same<typename Traits::local_ordinal_type, void>::value;
  static constexpr bool have_go =
    ! std::is_same<typename Traits::global_ordinal_type, void>::value;
  static constexpr bool have_nt =
    ! std::is_same<typename Traits::node_type, void>::value;

  // If we already have local_ordinal_type, don't try to get it again.
  // If we don't have local_ordinal_type and if First is an integer,
  // then use First as the local_ordinal_type.
  static constexpr bool use_lo = ! have_lo &&
    std::is_integral<First>::value;
  // If we already have global_ordinal_type, don't try to get it
  // again.  Also, if we're already using First as local_ordinal_type,
  // don't try to make First the global_ordinal_type.  Otherwise, if
  // First is an integer, make it the global_ordinal_type.
  static constexpr bool use_go = ! have_go && ! use_lo &&
    std::is_integral<First>::value;
  // If we already have node_type, don't try to get it again.  Also,
  // if we're already using First as local_ordinal_type or
  // global_ordinal_type, don't try to make First the node_type.
  // Otherwise, if First is "Device-like" and thus could be used to
  // construct node_type, do so.
  static constexpr bool use_nt = ! have_nt && ! use_lo && ! use_go &&
    is_device_like<First>::value;

  // The First template parameter must be usable.
  static_assert (use_lo || use_go || use_nt,
                 "The First template parameter is invalid.");
  static_assert (! use_nt || is_device_like<First>::value,
                 "Please report this bug to the Tpetra developers.");
  static_assert (! use_nt || ! std::is_integral<First>::value,
                 "Please report this bug to the Tpetra developers.");

  // Traits template parameter for the recursion.
  struct MyTraits {
    using local_ordinal_type =
      typename std::conditional<
        use_lo,
        First,
        typename Traits::local_ordinal_type
      >::type;
    using global_ordinal_type =
      typename std::conditional<
        use_go,
        First,
        typename Traits::global_ordinal_type
      >::type;
    using node_type =
      typename std::conditional<
        use_nt,
        typename NodeType<First>::type,
        typename Traits::node_type
      >::type;
  };

  // Recurse on the Rest of the list, using MyTraits as the Traits
  // template parameter.
  using rest_traits = ThreeArgTraitsImpl<MyTraits, Defaults, Rest...>;

public:
  using local_ordinal_type = typename rest_traits::local_ordinal_type;
  using global_ordinal_type = typename rest_traits::global_ordinal_type;
  using node_type = typename rest_traits::node_type;
};

// Recurse on the template parameter list.
template<class Traits,
         class Defaults,
         class First,
         class ... Rest>
struct FourArgTraitsImpl<Traits, Defaults, First, Rest...> {
private:
  static_assert (sizeof... (Rest) <= 4, "Tpetra classes like MultiVector "
                 "may have no more than four template parameters.");

  // Recursion depends on the aux Traits template parameter.  This is
  // always a struct with four typedefs: scalar_type,
  // local_ordinal_type, global_ordinal_type, and node_type.  The
  // typedefs start all as void.  We then fill them in as we learn
  // them.
  static constexpr bool have_sc =
    ! std::is_same<typename Traits::scalar_type, void>::value;
  static constexpr bool have_lo =
    ! std::is_same<typename Traits::local_ordinal_type, void>::value;
  static constexpr bool have_go =
    ! std::is_same<typename Traits::global_ordinal_type, void>::value;
  static constexpr bool have_nt =
    ! std::is_same<typename Traits::node_type, void>::value;

  // If we already have scalar_type, don't try to get it again.
  // Otherwise, if First is legit for scalar type, make it the
  // scalar_type.
  static constexpr bool use_sc = ! have_sc &&
    is_valid_scalar<First>::value;
  // If we already have local_ordinal_type, don't try to get it again.
  // Also, if we're already using First as scalar_type, don't try to
  // make First the local_ordinal_type.  Otherwise, if First is an
  // integer, make it the local_ordinal_type.
  static constexpr bool use_lo = ! have_lo &&
    ! use_sc &&
    std::is_integral<First>::value;
  // If we already have global_ordinal_type, don't try to get it
  // again.  Also, if we're already using First as scalar_type or
  // local_ordinal_type, don't try to make First the
  // global_ordinal_type.  Otherwise, if First is an integer, make it
  // the global_ordinal_type.
  static constexpr bool use_go = ! have_go &&
    ! use_sc && ! use_lo &&
    std::is_integral<First>::value;
  // If we already have node_type, don't try to get it again.  Also,
  // if we're already using First as scalar_type, local_ordinal_type,
  // or global_ordinal_type, don't try to make First the node_type.
  // Otherwise, if First is "Device-like" and thus could be used to
  // construct node_type, do so.
  static constexpr bool use_nt = ! have_nt &&
    ! use_lo && ! use_go && ! use_sc &&
    is_device_like<First>::value;
  // The First template parameter must be usable.
  static_assert (use_sc || use_lo || use_go || use_nt,
                 "The First template parameter is invalid.");

  // Traits template parameter for the recursion.
  struct MyTraits {
    using scalar_type =
      typename std::conditional<
        use_sc,
        First,
        typename Traits::scalar_type
      >::type;
    using local_ordinal_type =
      typename std::conditional<
        use_lo,
        First,
        typename Traits::local_ordinal_type
      >::type;
    using global_ordinal_type =
      typename std::conditional<
        use_go,
        First,
        typename Traits::global_ordinal_type
      >::type;
    using node_type =
      typename std::conditional<
        use_nt,
        typename NodeType<First>::type,
        typename Traits::node_type
      >::type;
  };

  // Recurse on the Rest of the list, using MyTraits as the Traits
  // template parameter.
  using rest_traits = FourArgTraitsImpl<MyTraits, Defaults, Rest...>;

public:
  using scalar_type = typename rest_traits::scalar_type;
  using local_ordinal_type = typename rest_traits::local_ordinal_type;
  using global_ordinal_type = typename rest_traits::global_ordinal_type;
  using node_type = typename rest_traits::node_type;
};

// Use ThreeArgTraitsImpl to translate a list of template parameters
// Args... into local_ordinal_type, global_ordinal_type, and node_type
// typedefs.  Get defaults for these three types from the first
// template parameter.
template<class Defaults, class ... Args>
struct ThreeArgTraits {
private:
  // ThreeArgTraitsImpl starts with a struct whose typedefs are all void.
  struct MyTraits {
    using local_ordinal_type = void;
    using global_ordinal_type = void;
    using node_type = void;
  };

public:
  using local_ordinal_type =
    typename ThreeArgTraitsImpl<MyTraits, Defaults, Args...>::local_ordinal_type;
  using global_ordinal_type =
    typename ThreeArgTraitsImpl<MyTraits, Defaults, Args...>::global_ordinal_type;
  using node_type =
    typename ThreeArgTraitsImpl<MyTraits, Defaults, Args...>::node_type;
};

// Use FourArgTraitsImpl to translate a list of template parameters
// Args... into scalar_type, local_ordinal_type, global_ordinal_type,
// and node_type typedefs.  Get defaults for these four types from
// the first template parameter.
template<class Defaults, class ... Args>
struct FourArgTraits {
private:
  // FourArgTraitsImpl starts with a struct whose typedefs are all void.
  struct MyTraits {
    using scalar_type = void;
    using local_ordinal_type = void;
    using global_ordinal_type = void;
    using node_type = void;
  };

public:
  using scalar_type =
    typename FourArgTraitsImpl<MyTraits, Defaults, Args...>::scalar_type;
  using local_ordinal_type =
    typename FourArgTraitsImpl<MyTraits, Defaults, Args...>::local_ordinal_type;
  using global_ordinal_type =
    typename FourArgTraitsImpl<MyTraits, Defaults, Args...>::global_ordinal_type;
  using node_type =
    typename FourArgTraitsImpl<MyTraits, Defaults, Args...>::node_type;
};

// Implementation detail.  Tpetra will use ThreeArgAlias to get the
// Classes::${CLASS} specialization corresponding to the template
// parameters of ${CLASS}.  First template parameter will be
// Classes::${CLASS}.  Get defaults for those template parameters from
// Defaults.  Remaining template parameter(s) (if any) Args are
// exactly the template parameters of ${CLASS}.
template<template <class LO, class GO, class NT> class Object,
         class Defaults,
         class ... Args>
struct ThreeArgAlias {
private:
  static_assert (sizeof... (Args) <= 3, "Users must supply no more than 3 "
                 "template parameters for Tpetra classes like Map "
                 "and CrsGraph.");
  using local_ordinal_type = typename ThreeArgTraits<Defaults, Args...>::local_ordinal_type;
  using global_ordinal_type = typename ThreeArgTraits<Defaults, Args...>::global_ordinal_type;
  using node_type = typename ThreeArgTraits<Defaults, Args...>::node_type;
public:
  using type = Object<local_ordinal_type, global_ordinal_type, node_type>;
};

// Implementation detail.  Tpetra will use FourArgAlias to get the
// Classes::MultiVector (or other Tpetra class with four template
// parameters) specialization corresponding to the template parameters
// of Map.  First template parameter will be Classes::${CLASSES}.  Get
// defaults for those template parameters from Defaults.  Remaining
// template parameter(s) (if any) Args are exactly the template
// parameters of ${CLASSES}.
template<template <class SC, class LO, class GO, class NT> class Object,
         class Defaults,
         class ... Args>
struct FourArgAlias {
private:
  static_assert (sizeof... (Args) <= 4, "Users must supply no more than 4 "
                 "template parameters for Tpetra classes like MultiVector "
                 "and CrsMatrix.");
  using scalar_type = typename FourArgTraits<Defaults, Args...>::scalar_type;
  using local_ordinal_type = typename FourArgTraits<Defaults, Args...>::local_ordinal_type;
  using global_ordinal_type = typename FourArgTraits<Defaults, Args...>::global_ordinal_type;
  using node_type = typename FourArgTraits<Defaults, Args...>::node_type;
public:
  using type = Object<scalar_type, local_ordinal_type, global_ordinal_type, node_type>;
};

} // namespace Details
} // namespace Tpetra

#endif // TPETRA_DETAILS_ALIASES_HPP
