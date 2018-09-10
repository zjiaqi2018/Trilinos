// @HEADER
// ************************************************************************
//
//                           Intrepid2 Package
//                 Copyright (2007) Sandia Corporation
//
// Under terms of Contract DE-AC04-94AL85000, there is a non-exclusive
// license for use of this work by or on behalf of the U.S. Government.
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
// Questions? Contact Kyungjoo Kim  (kyukim@sandia.gov), or
//                    Mauro Perego  (mperego@sandia.gov)
//
// ************************************************************************
// @HEADER

/** \file   Intrepid2_HGRAD_WEDGE_COMP12_FEMDef.hpp
    \brief  Definition file for FEM basis functions of degree 1 for H(grad)
   functions on WEDGE cells. \author Created by P. Bochev and D. Ridzal.
            Kokkorized by Kyungjoo Kim
*/

#ifndef __INTREPID2_HGRAD_WEDGE_COMP12_FEM_DEF_HPP__
#define __INTREPID2_HGRAD_WEDGE_COMP12_FEM_DEF_HPP__

namespace Intrepid2 {

// Map of numbering convention between Intrepid2's DOF numbering
// and parent triangle numbering for ease of reference (A=10, B=11)
// DOF   NODE
//                    BOTTOM PARENT TRIANGLE
//   0      2              0         2
//   1      0             / \       / \
//   2      1            6---8     5---4
//   3      2           / \ / \   / \ / \
//   4      0          1---7---2 0---3---1
//   5      1
//   6      5          TOP PARENT TRIANGLE
//   7      3              3         2
//   8      4             / \       / \
//   9      5            9---B     5---4
//   A      3           / \ / \   / \ / \
//   B      4          4---A---5 0---3---1

namespace Impl {

namespace {

template <typename ST>
KOKKOS_INLINE_FUNCTION
ordinal_type
getSubTriangle(ST const r, ST const s)
{
  auto const t = 1.0 - r - s;

  // Subtriangle E0
  if (0.5 <= r && r <= 1.0) return 0;

  // Subtriangle E1
  if (0.5 <= s && s <= 1.0) return 1;

  // Subtriangle E2
  if (0.5 <= t && t <= 1.0) return 2;

  // Subtriangle E3
  if ((0.0 <= r && r <= 0.5) && (0.0 <= s && s <= 0.5) &&
      (0.0 <= t && t <= 0.5))
    return 3;

  // Not in any of the subtriangles
  return -1;
}

template <typename ST>
struct Values
{
  ST N[12];
};

template <typename ST>
KOKKOS_INLINE_FUNCTION
Values<ST>
getInPlaneValues(ST const r, ST const s)
{
  auto const t = 1.0 - r - s;
  auto const subtriangle = getSubTriangle(r, s);
  ST N[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

  switch (subtriangle) {
  case 0:
    N[0] = 2.0 * r - 1.0;
    N[3] = 2.0 * s;
    N[5] = 2.0 * t;
    break;
  case 1:
    N[1] = 2.0 * s - 1.0;
    N[3] = 2.0 * r;
    N[4] = 2.0 * t;
    break;
  case 2:
    N[2] = 2.0 * t - 1.0;
    N[4] = 2.0 * s;
    N[5] = 2.0 * r;
    break;
  case 3:
    N[3] = 1.0 - 2.0 * t;
    N[4] = 1.0 - 2.0 * r;
    N[5] = 1.0 - 2.0 * s;
    break;
  default:
    // Outside parent triangle return zeros. No extrapolation.
    break;
  }

  ordinal_type I[12] = {2, 0, 1, 2, 0, 1, 5, 3, 4, 5, 3, 4};
  Values<ST> parallel;
  for (auto i = 0; i < 12; ++i) parallel.N[i] = N[I[i]];
  return parallel;
}

template <typename ST>
KOKKOS_INLINE_FUNCTION
Values<ST>
getOutPlaneValues(ST const z)
{
  ordinal_type I[12] = {0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1};
  ST Z[2] = {0.5 - 0.5 * z, 0.5 + 0.5 * z};
  Values<ST> perpendicular;
  for (auto i = 0; i < 12; ++i) perpendicular.N[i] = Z[I[i]];
  return perpendicular;
}

template <typename ST>
struct ParallDeriv
{
  ST dN[12][2];
};

template <typename ST>
KOKKOS_INLINE_FUNCTION
ParallDeriv<ST>
getInPlaneDerivatives(ST const r, ST const s)
{
  auto const subtriangle = getSubTriangle(r, s);
  ST dN[6][2] = {{0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0},
      {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}};

  switch (subtriangle) {
  case 0:
    dN[0][0] = 2.0;
    dN[3][1] = 2.0;
    dN[5][0] = -2.0;
    dN[5][1] = -2.0;
    break;
  case 1:
    dN[1][1] = 2.0;
    dN[3][0] = 2.0;
    dN[4][0] = -2.0;
    dN[4][1] = -2.0;
    break;
  case 2:
    dN[2][0] = -2.0;
    dN[2][1] = -2.0;
    dN[4][1] = 2.0;
    dN[5][0] = 2.0;
    break;
  case 3:
    dN[3][0] = 2.0;
    dN[3][1] = 2.0;
    dN[4][0] = -2.0;
    dN[5][1] = -2.0;
    break;
  default:
    // Outside parent triangle return zeros. No extrapolation.
    break;
  }

  ordinal_type I[12] = {2, 0, 1, 2, 0, 1, 5, 3, 4, 5, 3, 4};
  ParallDeriv<ST> parallel;

  for (auto i = 0; i < 12; ++i) {
    for (auto j = 0; j < 2; ++j) {
      parallel.dN[i][j] = dN[I[i]][j];
    }
  }

  return parallel;
}

template <typename ST>
struct PerpenDeriv
{
  ST dN[12][1];
};

template <typename ST>
KOKKOS_INLINE_FUNCTION
PerpenDeriv<ST>
getOutPlaneDerivatives(ST const z)
{
  ordinal_type I[12] = {0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1};
  ST dZ[2] = {-0.5, 0.5};
  PerpenDeriv<ST> perpendicular;
  for (auto i = 0; i < 12; ++i) perpendicular.dN[i][0] = dZ[I[i]];
  return perpendicular;
}

} // anonymous namespace

template <EOperator opType>
template <typename outputViewType, typename inputViewType>
KOKKOS_INLINE_FUNCTION void
Basis_HGRAD_WEDGE_COMP12_FEM::Serial<opType>::getValues(
    outputViewType      output,
    inputViewType const input)
{
  using ST = typename outputViewType::value_type;

  switch (opType) {
    case OPERATOR_VALUE: {
      auto const r = input(0);
      auto const s = input(1);
      auto const z = input(2);

      Values<ST> XY = getInPlaneValues(r, s);
      Values<ST> Z = getOutPlaneValues(z);

      for (auto i = 0; i < 12; ++i) output.access(i) = XY.N[i] * Z.N[i];

      break;
    }
    case OPERATOR_GRAD: {
      auto const r = input(0);
      auto const s = input(1);
      auto const z = input(2);

      Values<ST> XY = getInPlaneValues(r, s);
      Values<ST> Z = getOutPlaneValues(z);
      ParallDeriv<ST> dXY = getInPlaneDerivatives(r, s);
      PerpenDeriv<ST> dZ = getOutPlaneDerivatives(z);

      for (auto i = 0; i < 12; ++i) {
        output.access(i, 0) = dXY.dN[i][0] * Z.N[i];
        output.access(i, 1) = dXY.dN[i][1] * Z.N[i];
        output.access(i, 2) = XY.N[i] * dZ.dN[i][0];
      }

      break;
    }
    case OPERATOR_MAX: {
      const ordinal_type jend = output.extent(1);
      const ordinal_type iend = output.extent(0);

      for (ordinal_type j = 0; j < jend; ++j)
        for (auto i = 0; i < iend; ++i) output.access(i, j) = 0.0;
      break;
    }
    default: {
      INTREPID2_TEST_FOR_ABORT(
          true,
          ">>> ERROR (Basis_HGRAD_TET_COMP12_FEM): Operator type not "
          "implemented");
    }
  }
}

template <
    typename SpT,
    typename outputValueValueType,
    class... outputValueProperties,
    typename inputPointValueType,
    class... inputPointProperties>
void
Basis_HGRAD_WEDGE_COMP12_FEM::getValues(
    Kokkos::DynRankView<outputValueValueType, outputValueProperties...>
        outputValues,
    const Kokkos::DynRankView<inputPointValueType, inputPointProperties...>
                    inputPoints,
    const EOperator operatorType)
{
  typedef Kokkos::DynRankView<outputValueValueType, outputValueProperties...>
      outputValueViewType;
  typedef Kokkos::DynRankView<inputPointValueType, inputPointProperties...>
      inputPointViewType;
  typedef
      typename ExecSpace<typename inputPointViewType::execution_space, SpT>::
          ExecSpaceType ExecSpaceType;

  // Number of evaluation points = dim 0 of inputPoints
  auto const loopSize = inputPoints.extent(0);
  Kokkos::RangePolicy<ExecSpaceType, Kokkos::Schedule<Kokkos::Static>> policy(
      0, loopSize);

  switch (operatorType) {
    case OPERATOR_VALUE: {
      typedef Functor<outputValueViewType, inputPointViewType, OPERATOR_VALUE>
          FunctorType;
      Kokkos::parallel_for(policy, FunctorType(outputValues, inputPoints));
      break;
    }
    case OPERATOR_GRAD:
    case OPERATOR_D1: {
      typedef Functor<outputValueViewType, inputPointViewType, OPERATOR_GRAD>
          FunctorType;
      Kokkos::parallel_for(policy, FunctorType(outputValues, inputPoints));
      break;
    }
    case OPERATOR_CURL: {
      INTREPID2_TEST_FOR_EXCEPTION(
          operatorType == OPERATOR_CURL,
          std::invalid_argument,
          ">>> ERROR (Basis_HGRAD_WEDGE_COMP12_FEM): CURL is invalid operator "
          "for rank-0 (scalar) functions in 3D");
      break;
    }

    case OPERATOR_DIV: {
      INTREPID2_TEST_FOR_EXCEPTION(
          (operatorType == OPERATOR_DIV),
          std::invalid_argument,
          ">>> ERROR (Basis_HGRAD_WEDGE_COMP12_FEM): DIV is invalid operator "
          "for rank-0 (scalar) functions in 3D");
      break;
    }

    case OPERATOR_D2: {
      typedef Functor<outputValueViewType, inputPointViewType, OPERATOR_D2>
          FunctorType;
      Kokkos::parallel_for(policy, FunctorType(outputValues, inputPoints));
      break;
    }
    case OPERATOR_D3:
    case OPERATOR_D4:
    case OPERATOR_D5:
    case OPERATOR_D6:
    case OPERATOR_D7:
    case OPERATOR_D8:
    case OPERATOR_D9:
    case OPERATOR_D10: {
      typedef Functor<outputValueViewType, inputPointViewType, OPERATOR_MAX>
          FunctorType;
      Kokkos::parallel_for(policy, FunctorType(outputValues, inputPoints));
      break;
    }
    default: {
      INTREPID2_TEST_FOR_EXCEPTION(
          !(Intrepid2::isValidOperator(operatorType)),
          std::invalid_argument,
          ">>> ERROR (Basis_HGRAD_WEDGE_COMP12_FEM): Invalid operator type");
    }
  }
}
}  // namespace Impl
// -------------------------------------------------------------------------------------

template <typename SpT, typename OT, typename PT>
Basis_HGRAD_WEDGE_COMP12_FEM<SpT, OT, PT>::Basis_HGRAD_WEDGE_COMP12_FEM()
{
  this->basisCardinality_ = 6;
  this->basisDegree_      = 1;
  this->basisCellTopology_ =
      shards::CellTopology(shards::getCellTopologyData<shards::Wedge<6>>());
  this->basisType_        = BASIS_FEM_DEFAULT;
  this->basisCoordinates_ = COORDINATES_CARTESIAN;

  // initialize tags
  {
    // Basis-dependent intializations
    const ordinal_type tagSize = 4;  // size of DoF tag
    const ordinal_type posScDim =
        0;  // position in the tag, counting from 0, of the subcell dim
    const ordinal_type posScOrd =
        1;  // position in the tag, counting from 0, of the subcell ordinal
    const ordinal_type posDfOrd = 2;  // position in the tag, counting from 0,
                                      // of DoF ordinal relative to the subcell

    // An array with local DoF tags assigned to basis functions, in the order of
    // their local enumeration
    ordinal_type tags[24] = {0, 0, 0, 1, 0, 1, 0, 1, 0, 2, 0, 1,
                             0, 3, 0, 1, 0, 4, 0, 1, 0, 5, 0, 1};

    // host tags
    ordinal_type_array_1d_host tagView(&tags[0], 24);

    // Basis-independent function sets tag and enum data in tagToOrdinal_ and
    // ordinalToTag_ arrays:
    // ordinal_type_array_2d_host ordinalToTag;
    // ordinal_type_array_3d_host tagToOrdinal;
    this->setOrdinalTagData(
        this->tagToOrdinal_,
        this->ordinalToTag_,
        tagView,
        this->basisCardinality_,
        tagSize,
        posScDim,
        posScOrd,
        posDfOrd);
  }

  // dofCoords on host and create its mirror view to device
  Kokkos::DynRankView<
      typename scalarViewType::value_type,
      typename SpT::array_layout,
      Kokkos::HostSpace>
      dofCoords(
          "dofCoordsHost",
          this->basisCardinality_,
          this->basisCellTopology_.getDimension());

  dofCoords(0, 0) = 0.0;
  dofCoords(0, 1) = 0.0;
  dofCoords(0, 2) = -1.0;
  dofCoords(1, 0) = 1.0;
  dofCoords(1, 1) = 0.0;
  dofCoords(1, 2) = -1.0;
  dofCoords(2, 0) = 0.0;
  dofCoords(2, 1) = 1.0;
  dofCoords(2, 2) = -1.0;
  dofCoords(3, 0) = 0.0;
  dofCoords(3, 1) = 0.0;
  dofCoords(3, 2) = 1.0;
  dofCoords(4, 0) = 1.0;
  dofCoords(4, 1) = 0.0;
  dofCoords(4, 2) = 1.0;
  dofCoords(5, 0) = 0.0;
  dofCoords(5, 1) = 1.0;
  dofCoords(5, 2) = 1.0;

  this->dofCoords_ =
      Kokkos::create_mirror_view(typename SpT::memory_space(), dofCoords);
  Kokkos::deep_copy(this->dofCoords_, dofCoords);
}

}  // namespace Intrepid2
#endif
