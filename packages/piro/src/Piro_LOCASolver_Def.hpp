// @HEADER
// ************************************************************************
//
//        Piro: Strategy package for embedded analysis capabilitites
//                  Copyright (2010) Sandia Corporation
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
// Questions? Contact Andy Salinger (agsalin@sandia.gov), Sandia
// National Laboratories.
//
// ************************************************************************
// @HEADER

#ifndef PIRO_LOCASOLVER_DEF_HPP
#define PIRO_LOCASOLVER_DEF_HPP

#include "Piro_LOCASolver.hpp"

#include "Piro_ObserverToLOCASaveDataStrategyAdapter.hpp"

#include "Thyra_DetachedVectorView.hpp"

#include "NOX_StatusTest_Factory.H"

#include "Piro_MatrixFreeDecorator.hpp" 

#include "Teuchos_as.hpp"
#include "Teuchos_TestForException.hpp"
#include "Teuchos_Assert.hpp"

#include "NOX_PrePostOperator_Vector.H"
#include "NOX_PrePostOperator_RowSumScaling.H"
#include "NOX_MeritFunction_Weighted.hpp"

#include <stdexcept>
#include <ostream>

namespace Piro {

namespace Detail {

class ModelEvaluatorParamName {
public:
  explicit ModelEvaluatorParamName(const Teuchos::RCP<const Teuchos::Array<std::string> > &p_names);
  std::string operator()(Teuchos_Ordinal k) const;

private:
  Teuchos::RCP<const Teuchos::Array<std::string> > p_names_;
  enum { Default, OneShared, FullList } type_;
};

} // namespace Detail

} // namespace Piro


template <typename Scalar>
Piro::LOCASolver<Scalar>::LOCASolver(
    const Teuchos::RCP<Teuchos::ParameterList> &piroParams,
    const Teuchos::RCP<Thyra::ModelEvaluator<Scalar> > &model,
    const Teuchos::RCP<LOCA::Thyra::SaveDataStrategy> &saveDataStrategy) :
  SteadyStateSolver<Scalar>(model, model->Np() > 0), // Only one parameter supported
  piroParams_(piroParams),
  saveDataStrategy_(saveDataStrategy),
  globalData_(LOCA::createGlobalData(piroParams)),
  paramVector_(),
  group_(),
  locaStatusTests_(),
  noxStatusTests_(),
  stepper_(),
  model_(model)
{
  std::cout << "IKT piroParams = " << *piroParams << "\n"; 
  const int l = 0; // TODO: Allow user to select parameter index
  const Detail::ModelEvaluatorParamName paramName(this->getModel().get_p_names(l));
  const Thyra::Ordinal p_entry_count = this->getModel().get_p_space(l)->dim();
  for (Teuchos_Ordinal k = 0; k < p_entry_count; ++k) {
    (void) paramVector_.addParameter(paramName(k));
  }
  
  std::string jacobianSource = piroParams->get("Jacobian Operator", "Have Jacobian");
  if (jacobianSource == "Matrix-Free") {
    if (piroParams->isParameter("Matrix-Free Perturbation")) {
      model_ = Teuchos::rcp(new Piro::MatrixFreeDecorator<Scalar>(model,
                           piroParams->get<double>("Matrix-Free Perturbation")));
    }
    else model_ = Teuchos::rcp(new Piro::MatrixFreeDecorator<Scalar>(model));
  }

  const NOX::Thyra::Vector initialGuess(*model_->getNominalValues().get_x());

  // Get NOX->Thyra Group Options sublist and set params relevant to row sum scaling
  const Teuchos::RCP<Teuchos::ParameterList> thyra_group_options_sublist =
    Teuchos::sublist(Teuchos::sublist(piroParams_, "NOX"), "Thyra Group Options");
  std::string string_when_to_update = thyra_group_options_sublist->get<std::string>("Update Row Sum Scaling"); 
  if (string_when_to_update == "Before Each Nonlinear Solve")
    when_to_update_ = NOX::RowSumScaling::UpdateInvRowSumVectorAtBeginningOfSolve;
  else if (string_when_to_update == "Before Each Nonlinear Iteration")
    when_to_update_ = NOX::RowSumScaling::UpdateInvRowSumVectorAtBeginningOfIteration;
  function_scaling_ = thyra_group_options_sublist->get<std::string>("Function Scaling");
  if (function_scaling_ =="Row Sum")
    do_row_sum_scaling_ = true;
  else
    do_row_sum_scaling_ = false;
  std::cout << "IKT function_scaling_ = " << function_scaling_ << "\n"; 
  std::cout << "IKT string_when_to_update = " << string_when_to_update << "\n"; 

  Teuchos::RCP<Thyra::VectorBase<Scalar>> scaling_vector;  
  if (function_scaling_ == "None") {
    std::cout << "IKT no scaling\n"; 
    scaling_vector = Teuchos::null;
  }
  else {
    std::cout << "IKT scaling\n"; 
    if (do_row_sum_scaling_ == true) {
      std::cout << "IKT row sum scaling \n"; 
      scaling_vector =  setupRowSumScalingObjects(); 
    }
    //IKT FIXME: throw exception otherwise 
  }
  //Setup Row Sum Scaling objects 
  //Create LOCA::Thyra::Group 
  group_ = Teuchos::rcp(new LOCA::Thyra::Group(globalData_, initialGuess, model_, paramVector_, l, false, scaling_vector));
  group_->setSaveDataStrategy(saveDataStrategy_);

  // TODO: Create non-trivial stopping criterion for the stepper
  locaStatusTests_ = Teuchos::null;

  // Create stopping criterion for the nonlinear solver
  const Teuchos::RCP<Teuchos::ParameterList> noxStatusParams =
    Teuchos::sublist(Teuchos::sublist(piroParams_, "NOX"), "Status Tests");
  noxStatusTests_ = NOX::StatusTest::buildStatusTests(*noxStatusParams, *(globalData_->locaUtils));

  std::cout << "IKT piroParams modified = " << *piroParams_ << "\n";  

  stepper_ = Teuchos::rcp(new LOCA::Stepper(globalData_, group_, locaStatusTests_, noxStatusTests_, piroParams_));
  first_ = true;

  if (piroParams_->isSublist("NOX") &&
      piroParams_->sublist("NOX").isSublist("Printing"))
    utils_.reset(piroParams_->sublist("NOX").sublist("Printing"));
}

template<typename Scalar>
Piro::LOCASolver<Scalar>::~LOCASolver()
{
  LOCA::destroyGlobalData(globalData_);
}

template<typename Scalar>
Teuchos::RCP<NOX::Solver::Generic>
Piro::LOCASolver<Scalar>::getSolver()
{
  return stepper_->getSolver();
}

template<typename Scalar>
Teuchos::ParameterList &
Piro::LOCASolver<Scalar>::getStepperParams()
{
  return stepper_->getParams();
}

template<typename Scalar>
Teuchos::ParameterList &
Piro::LOCASolver<Scalar>::getStepSizeParams()
{
  return stepper_->getStepSizeParams();
}

template <typename Scalar>
void
Piro::LOCASolver<Scalar>::evalModelImpl(
    const Thyra::ModelEvaluatorBase::InArgs<Scalar>& inArgs,
    const Thyra::ModelEvaluatorBase::OutArgs<Scalar>& outArgs) const
{
  std::cout << "IKT Piro::LOCASolver::evalModelImpl\n"; 
  const int l = 0; // TODO: Allow user to select parameter index
  const Teuchos::RCP<const Thyra::VectorBase<Scalar> > p_inargs = inArgs.get_p(l);

  // Forward parameter values to the LOCA stepper
  {
    const Teuchos::RCP<const Thyra::VectorBase<Scalar> > p_inargs_or_nominal =
      Teuchos::nonnull(p_inargs) ? p_inargs : this->getNominalValues().get_p(l);
    const Thyra::ConstDetachedVectorView<Scalar> p_init_values(p_inargs_or_nominal);
    const Teuchos_Ordinal p_entry_count = p_init_values.subDim();
    TEUCHOS_ASSERT(p_entry_count == Teuchos::as<Teuchos_Ordinal>(paramVector_.length()));

    for (Teuchos_Ordinal k = 0; k < p_entry_count; ++k) {
      paramVector_[k] = p_init_values[k];
    }

    group_->setParams(paramVector_);
  }

   
  if (first_) {
    // No need to call reset. The call can result in long stdout output, so it's
    // nice to avoid it since we can.
    first_ = false;
  } else
    stepper_->reset(globalData_, group_, locaStatusTests_, noxStatusTests_, piroParams_);
  const LOCA::Abstract::Iterator::IteratorStatus status = stepper_->run();

  if (status == LOCA::Abstract::Iterator::Finished) {
    utils_.out() << "Continuation Stepper Finished.\n";;
  } else if (status == LOCA::Abstract::Iterator::NotFinished) {
    utils_.out() << "Continuation Stepper did not reach final value.\n";
  } else {
    utils_.out() << "Nonlinear solver failed to converge.\n";
    outArgs.setFailed();
  }

  const Teuchos::RCP<Thyra::VectorBase<Scalar> > x_outargs = outArgs.get_g(this->num_g());
  const Teuchos::RCP<Thyra::VectorBase<Scalar> > x_final =
    Teuchos::nonnull(x_outargs) ? x_outargs : Thyra::createMember(this->get_g_space(this->num_g()));

  {
    // Deep copy final solution from LOCA group
    NOX::Thyra::Vector finalSolution(x_final);
    finalSolution = group_->getX();
  }

  // Compute responses for the final solution
  {
    Thyra::ModelEvaluatorBase::InArgs<Scalar> modelInArgs =
      this->getModel().createInArgs();
    {
      modelInArgs.set_x(x_final);
      modelInArgs.set_p(l, p_inargs);
    }

    this->evalConvergedModel(modelInArgs, outArgs);
  }
}

template <typename Scalar>
Teuchos::RCP<Thyra::VectorBase<Scalar> > 
Piro::LOCASolver<Scalar>::setupRowSumScalingObjects() const 
{
  std::cout << "IKT Piro::LOCASolver::setupRowSumScalingObjects\n"; 
  Teuchos::RCP<Thyra::VectorBase<Scalar>> scaling_vector =  ::Thyra::createMember(model_->get_f_space());
  std::cout << "IKT scaling_vector = " << scaling_vector << "\n"; 
  ::Thyra::V_S(scaling_vector.ptr(),1.0);
  Teuchos::RCP<NOX::Abstract::PrePostOperator> row_sum_observer =
      Teuchos::rcp(new NOX::RowSumScaling(scaling_vector, when_to_update_));

  Teuchos::ParameterList& nox_params = piroParams_->sublist("NOX");  

  if (nox_params.sublist("Solver Options").
    isType< Teuchos::RCP<NOX::Abstract::PrePostOperator> >("User Defined Pre/Post Operator")) {
    std::cout << "IKT here1\n"; 

    Teuchos::RCP<NOX::Abstract::PrePostOperator> user_observer =
      nox_params.sublist("Solver Options").get< Teuchos::RCP<NOX::Abstract::PrePostOperator> >("User Defined Pre/Post Operator");

    // NOTE: the row_sum_observer should be evalauted after any user
    // oberservers to make sure that the jacobian is accurate.  This
    // is needed, for example, if we have a model evaluator decorator
    // that adds extra input parameters to the model such as a
    // predictor or previous time step solution to be used for
    // semi-implicit models.  The row sum would accidentally use the
    // previous predicted value which would be bad.
    Teuchos::RCP<NOX::PrePostOperatorVector> observer_vector = Teuchos::rcp(new NOX::PrePostOperatorVector);
    observer_vector->pushBack(user_observer);
    observer_vector->pushBack(row_sum_observer);

    nox_params.sublist("Solver Options").set< Teuchos::RCP<NOX::Abstract::PrePostOperator> >("User Defined Pre/Post Operator", observer_vector);
  }
  else {
    std::cout << "IKT here2\n"; 
    nox_params.sublist("Solver Options").set< Teuchos::RCP<NOX::Abstract::PrePostOperator> >("User Defined Pre/Post Operator", row_sum_observer);
  }

  // Set the weighted merit function.  Throw error if a user defined
  // merit funciton is present.
  // ETP 5/23/16 -- Commenting this out because the parameter list may have
  // been reused from previous solves, and so the merit function that has been
  // set below from previous solves may still be there.
  //TEUCHOS_ASSERT( !(nox_params.sublist("Solver Options").isType<Teuchos::RCP<NOX::MeritFunction::Generic> >("User Defined Merit Function")));

  Teuchos::RCP<NOX::MeritFunction::Generic> mf = Teuchos::rcp(new NOX::Thyra::WeightedMeritFunction(scaling_vector));
  nox_params.sublist("Solver Options").set<Teuchos::RCP<NOX::MeritFunction::Generic> >("User Defined Merit Function",mf);
  return scaling_vector; 
}

template <typename Scalar>
Teuchos::RCP<Piro::LOCASolver<Scalar> >
Piro::observedLocaSolver(
    const Teuchos::RCP<Teuchos::ParameterList> &appParams,
    const Teuchos::RCP<Thyra::ModelEvaluator<Scalar> > &model,
    const Teuchos::RCP<Piro::ObserverBase<Scalar> > &observer)
{
  const Teuchos::RCP<LOCA::Thyra::SaveDataStrategy> saveDataStrategy =
    Teuchos::nonnull(observer) ?
    Teuchos::rcp(new Piro::ObserverToLOCASaveDataStrategyAdapter(observer)) :
    Teuchos::null;

  return Teuchos::rcp(new Piro::LOCASolver<Scalar>(appParams, model, saveDataStrategy));
}



#endif /* PIRO_LOCASOLVER_DEF_HPP */
