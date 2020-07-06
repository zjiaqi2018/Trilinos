#!/bin/bash -l

if [ "${Trilinos_CTEST_DO_ALL_AT_ONCE}" == "" ] ; then
  export Trilinos_CTEST_DO_ALL_AT_ONCE=TRUE
fi

if [[ "${Trilinos_ENABLE_BUILD_STATS}" == "" ]] && \
   [[ ! $JOB_NAME == *"intel"* ]] \
  ; then
  export Trilinos_ENABLE_BUILD_STATS=ON
fi
echo "Trilinos_ENABLE_BUILD_STATS='${Trilinos_ENABLE_BUILD_STATS}'"

set -x

$WORKSPACE/Trilinos/cmake/ctest/drivers/atdm/ctest-s-driver.sh
