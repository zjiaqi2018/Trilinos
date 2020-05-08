// @HEADER
// ****************************************************************************
//                Tempus: Copyright (2017) Sandia Corporation
//
// Distributed under BSD 3-clause license (See accompanying file Copyright.txt)
// ****************************************************************************
// @HEADER

#ifndef Tempus_StepperLeapfrogAppAction_hpp
#define Tempus_StepperLeapfrogAppAction_hpp

#include "Tempus_config.hpp"
#include "Tempus_SolutionHistory.hpp"
#include "Tempus_StepperLeapfrog.hpp"


namespace Tempus {

// Forward Declaration for recursive includes (this AppAction <--> Stepper)
template<class Scalar> class StepperLeapfrog;

/** \brief Application Action for StepperLeapfrog.
 *
 *  This class provides a means to apply various actions with the Leapfrog time step.
 *  The data available to this class is solution variables (through
 *  SolutionHistory), and stepper data (through the Stepper).  It allows
 *  the application to just observe this data (i.e., use but not change the
 *  data) to change any of it (USER BEWARE!).
 *
 *  Below is the Leapfrog algorithm and includes the locations where the
 *  application can take actions (in italicized).
 *
 *  \f{algorithm}{
 *  \renewcommand{\thealgorithm}{}
 *  \caption{Leapfrog with the locations of the application actions indicated.}
 *  \begin{algorithmic}[1]
 *   Leapfrog
 *  \end{algorithmic}
 *  \f}
 */
template<class Scalar>
class StepperLeapfrogAppAction
{
public:

  /// Indicates the location of application action (see algorithm).
  enum ACTION_LOCATION {
    BEGIN_STEP,     ///< At the beginning of the step.                                     
    BEFORE_XDOT_UPDATE_INITIALIZE, // Before updating xDot while initializing xDotDot      
    BEFORE_X_UPDATE, //  Before updating x                                                 
    BEFORE_EXPLICIT_EVAL,   /// Before the explicit ME evaluation.                         
    BEFORE_XDOT_UPDATE, /// Before updating xDot                                           
    END_STEP        ///< At the end of the step.    
  };

  /// Constructor
  StepperLeapfrogAppAction(){}

  /// Destructor
  virtual ~StepperLeapfrogAppAction(){}

  /// Execute application action for Leapfrog Stepper.
  virtual void execute(
    Teuchos::RCP<SolutionHistory<Scalar> > sh,
    Teuchos::RCP<StepperLeapfrog<Scalar> > stepper,
    const typename StepperLeapfrogAppAction<Scalar>::ACTION_LOCATION actLoc) = 0;
};

} // namespace Tempus

#endif // Tempus_StepperLeapfrogAppAction_hpp
