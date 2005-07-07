#!/bin/csh
# ************************************************************************
# 
#                 Amesos
#                 Copyright (2004) Sandia Corporation
# 
# Under terms of Contract DE-AC04-94AL85000, there is a non-exclusive
# license for use of this work by or on behalf of the U.S. Government.
# 
# This library is free software; you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as
# published by the Free Software Foundation; either version 2.1 of the
# License, or (at your option) any later version.
#  
# This library is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.
#  
# You should have received a copy of the GNU Lesser General Public
# License along with this library; if not, write to the Free Software
# Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307
# USA
# Questions? Ken Stanley kstanley@cs.utk.edu or Marzio Sala msala@sandia.gov 
# 
# ************************************************************************
#
#  TestQuietAmesos improves upon TestAmesos by requiring that the tests 
#  (and by implication, the underlying Amesos class functions) be quiet - 
#  perform no prints.  
#
## NOTE: Those wishing to cusomize this script to run test exe's
## that have already been autotool'ed should read lines beginning with '##'

# $1 - Used only for automated testing.  No action required by script owner.
#      This parameter names the subdirectory from which the script is being
#      run.  This assists a developer in figuring out which tests failed.
# $2 - Indicates if the test is an automated nightly test.  No action required
#      by script owner.

## Some machines use a command different than mpirun to run mpi jobs.  The
## test-harness.plx script sets the environment variable
## "TRILINOS_TEST_HARNESS_MPIGO_COMMAND".  We test for
## this value below.  If not set, we set it to a default value.

set mpigo = `printenv TRILINOS_TEST_HARNESS_MPIGO_COMMAND`

if ("$mpigo" == "") then
    set mpigo = "mpirun -np "
endif

set error = None
set AnError = False
set printexitvalue
if( "$2" == "True" ) then # $2 is an optional parameter indicating if 
			  # this is an automated test or not
    # file2 is the log that is created and put into a results email if 
    # errors occur.
    set file2 = ../logMpiErrors.txt
    rm -f $file2
    # Echo some important information into the log file to help developers
    # figure out which tests failed.
    #'file' is a shorter log that is retained even if all tests pass.
    set file = ../log`eval uname`.txt
    rm -f $file
## IMPORTANT: Specify the script owner(s) on the following line
## For one owner type "owner@abc.com", for multiple owners
## "owner1@abc.com, owner2@def.com"
    #echo "msala@sandia.gov, amesos-regression@software.sandia.gov" >>& $file
    echo "Script owner(s) is listed on the previous line." >>& $file
## List the Trilinos package being tested on the following line
    echo "Package being tested: Amesos  " >>& $file
    echo "Name of subdirectory: " $1 >>& $file
    #  tempfile2 (file4) is used only in the creation of the longer file, $file2
    set file4 = tempfile2
endif
echo $file
echo $file2
echo "Date: " `eval date` >>& $file
echo `uname -a` >>& $file
## Different directory structures will require different setups.
## This file assumes a structure like that of epetra - exe's live in 
## a direct subdirectory of 'epetra/test' 

## Keep in mind that file and file2-4 live in 'package_name/test'
## Also, 'package_name/test' is the current directory
## It is recommended that all directory changing be done relative to
## the current directory because scripts live in the source directory,
## but are invoked from various build directories

## List the subdirectories of 'test' containing test exe's in the foreach loop
## if directory structure is like that of epetra.
#foreach f ( Test_Epetra_RowMatrix Test_Epetra_CrsMatrix Test_EpetraVbrMatrix Test_MultipleSolves Test_Detailed Test_LAPACK Test_KLU Test_UMFPACK Test_SuperLU Test_SuperLU_DIST Test_MUMPS Test_DSCPACK )
# Note: I check the failure or success using file Amesos_OK.
#       If this file is present, the test failed.
#       If this file does NOT exist, then the test completed successfully.
#       In fact, some mpi implemenations (e.g. LAM/MPI) returns 0
#       independently of the return status of the executable.
#
# FIXME: Test_MultipleSolves is not passed on all machines!
# foreach f ( Test_Epetra_RowMatrix Test_Epetra_CrsMatrix Test_Epetra_VbrMatrix Test_Detailed Test_UMFPACK Test_LAPACK Test_KLU Test_SuperLU Test_SuperLU_DIST Test_MUMPS Test_DSCPACK TestOptions )
#  Set MPI_GROUP_MAX to allow this test to pass desipte bug #1210
setenv MPI_COMM_MAX 4096
setenv MPI_GROUP_MAX 4096
foreach f ( Test_Epetra_RowMatrix Test_SuperLU_DIST TestOptions )
  cd $f
  set exefiles = (*.exe)
  set TestRan = False 
  if ( "${exefiles}X" != 'X' ) then
    foreach g(*.exe)
    set TestRan = True
      echo "" >>& ../$file
      echo "############" $g "##############" >>& ../$file
      if( "$2" == "True" ) then
        /bin/rm -f Amesos_FAILED
        /bin/rm -f ../$file4
        $mpigo 1 ./$g -q >>& ../$file4
#        /bin/rm -f ../$file4 ; touch ../$file4
        # ================== #
        # run with 1 process #		    
        # ================== #
        if( $status != 0 || -f Amesos_FAILED || ! -z ../$file4 ) then
          # A test failed.
          set AnError = True
	  if ( ! -z ../$file4 ) then 
             echo "  ******** Test w/ 4 proc yielded these messages: ********" >>& ../$file
	     cat ../$file4 >> ../$file 
             echo "  ******** End of messages ********" >>& ../$file
          endif
          echo "  ******** Test w/ 1 proc failed ********" >>& ../$file
          echo "Errors for script " $g " are listed above." >>& ../$file2
          echo "################### " $g " ##################" >>& ../$file2
	  $mpigo 1  ./$g -v >>& ../$file2
        else
          # Tests passed
          echo "******** Test w/ 1 proc passed ********" >>& ../$file
        endif
        # ==================== #
        # run with 4 processes #		    
        # ==================== #
        /bin/rm -f Amesos_FAILED
        $mpigo 4 ./$g -q >>& ../$file4
        if( $status != 0 || -f Amesos_FAILED  || ! -z ../$file4  ) then
          # A test failed.
          set AnError = True
	  if ( ! -z ../$file4 ) then 
             echo "  ******** Test w/ 4 proc yielded these messages: ********" >>& ../$file
	     cat ../$file4 >> ../$file 
             echo "  ******** End of messages ********" >>& ../$file
          endif
          echo "  ******** Test w/ 4 proc failed ********" >>& ../$file
          echo "Errors for script " $g " are listed above." >>& ../$file2
          echo "################### " $g " ##################" >>& ../$file2
	  $mpigo 4  ./$g -v >>& ../$file2
          else
          # Tests passed
          echo "******** Test w/ 4 proc passed ********" >>& ../$file
        endif
        /bin/rm -f Amesos_OK
      end
    endif
  endif
  if ( $TestRan != "True" ) then
    echo "No executables were found  " 
    set AnError = True
  endif
  cd ..
end

## At this point, it is assumed that the current directory is
## 'package_name/test'
if ( "$2" == "True" ) then
    echo "@#@#@#@#  Summary file @#@#@#@#@"
    cat $file
    rm $file
    if( "$AnError" != "True" ) then
	echo "@#@#@#@# Error file @#@#@#@#@"
	cat $file2
	rm -f $file2
    endif
endif

if ( "$AnError" == "True" ) then
    exit 1
else
    echo "End Result: TEST PASSED"
    exit 0
endif

