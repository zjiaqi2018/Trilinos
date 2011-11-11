// @HEADER
// ***********************************************************************
//
//         Zoltan2: Sandia Partitioning Ordering & Coloring Library
//
//                Copyright message goes here.   TODO
//
// ***********************************************************************
//
// Basic testing of Zoltan2::XpetraVectorInput 
//
//   TODO - we just look at the input adapter on stdout.  We should
//     check in code that it's correct.

#include <string>

#include <UserInputForTests.hpp>

#include <Zoltan2_XpetraVectorInput.hpp>
#include <Zoltan2_InputTraits.hpp>

#include <Teuchos_GlobalMPISession.hpp>
#include <Teuchos_DefaultComm.hpp>
#include <Teuchos_RCP.hpp>
#include <Teuchos_Comm.hpp>
#include <Teuchos_CommHelpers.hpp>

using namespace std;
using Teuchos::RCP;
using Teuchos::rcp;
using Teuchos::rcp_const_cast;
using Teuchos::Comm;
using Teuchos::DefaultComm;

typedef double scalar_t;
typedef int lno_t;
typedef int gno_t;
typedef Zoltan2::default_node_t node_t;

typedef UserInputForTests<scalar_t, lno_t, gno_t> uinput_t;
typedef Tpetra::Vector<scalar_t, lno_t, gno_t, node_t> tvector_t;
typedef Xpetra::Vector<scalar_t, lno_t, gno_t, node_t> xvector_t;
typedef Epetra_Vector evector_t;

int globalFail(RCP<const Comm<int> > &comm, int fail)
{
  int gfail=0;
  Teuchos::reduceAll<int,int>(*comm, Teuchos::REDUCE_SUM, 1, &fail, &gfail);
  return gfail;
}

void printFailureCode(RCP<const Comm<int> > &comm, int fail)
{
  int rank = comm->getRank();
  int nprocs = comm->getSize();
  comm->barrier();
  for (int p=0; p < nprocs; p++){
    if (p == rank)
      std::cout << rank << ": " << fail << std::endl;
    comm->barrier();
  }
  comm->barrier();
  if (rank==0) std::cout << "FAIL" << std::endl;
  exit(1);
}

template <typename S, typename L, typename G>
  void printVector(RCP<const Comm<int> > &comm, L nvtx,
    const G *vtxIds, const S *vals)
{
  int rank = comm->getRank();
  int nprocs = comm->getSize();
  comm->barrier();
  for (int p=0; p < nprocs; p++){
    if (p == rank){
      std::cout << rank << ":" << std::endl;
      for (L i=0; i < nvtx; i++){
        std::cout << " " << vtxIds[i] << ": " << vals[i] << std::endl;
      }
      std::cout.flush();
    }
    comm->barrier();
  }
  comm->barrier();
}

template <typename User>
int verifyInputAdapter(
  Zoltan2::XpetraVectorInput<User> &ia, tvector_t &vector)
{
  typedef typename Zoltan2::InputTraits<User>::scalar_t S;
  typedef typename Zoltan2::InputTraits<User>::lno_t L;
  typedef typename Zoltan2::InputTraits<User>::gno_t G;

  RCP<const Comm<int> > comm = vector.getMap()->getComm();
  int fail = 0, gfail=0;

  if (!ia.haveLocalIds())
    fail = 1;

  size_t base;
  if (!fail && !ia.haveConsecutiveLocalIds(base))
    fail = 2;

  if (!fail && base != 0)
    fail = 3;

  if (!fail && ia.getLocalLength() != vector.getLocalLength())
    fail = 4;

  if (!fail && ia.getGlobalLength() != vector.getGlobalLength())
    fail = 5;

  gfail = globalFail(comm, fail);

  const G *vtxIds=NULL;
  const L *lids=NULL;
  const S *vals=NULL;
  const S *wgts=NULL;
  size_t nvals=0;

  if (!gfail){

    nvals = ia.getVectorView(vtxIds, lids, vals, wgts);

    if (nvals != vector.getLocalLength())
      fail = 8;
    if (!fail && lids != NULL)   // implies consecutive
      fail = 9;
    if (!fail && wgts != NULL)   // not implemented yet
      fail = 10;

    gfail = globalFail(comm, fail);

    if (gfail == 0){
      printVector<S, L, G>(comm, nvals, vtxIds, vals);
    }
  }
  return fail;
}

int main(int argc, char *argv[])
{
  Teuchos::GlobalMPISession session(&argc, &argv);
  RCP<const Comm<int> > comm = DefaultComm<int>::getComm();
  int rank = comm->getRank();
  int nprocs = comm->getSize();
  int fail = 0, gfail=0;

  // Create object that can give us test Tpetra, Xpetra
  // and Epetra vectors for testing.

  RCP<uinput_t> uinput;

  try{
    uinput = 
      rcp(new uinput_t(std::string("../data/simple.mtx"), comm));
  }
  catch(std::exception &e){
    TEST_FAIL_AND_EXIT(*comm, 0, string("input ")+e.what(), 1);
  }

  RCP<tvector_t> tV;     // original vector (for checking)
  RCP<tvector_t> newV;   // migrated vector

  tV = uinput->getTpetraVector();
  size_t vlen = tV->getLocalLength();
  Teuchos::ArrayView<const gno_t> vtxGids = tV->getMap()->getNodeElementList();
  lno_t *vtxLids = NULL;

  /////////////////////////////////////////////////////////////
  // User object is Tpetra::Vector
  if (!gfail){ 
    RCP<const tvector_t> ctV = rcp_const_cast<const tvector_t>(tV);
    RCP<Zoltan2::XpetraVectorInput<tvector_t> > tVInput;
  
    try {
      tVInput = 
        rcp(new Zoltan2::XpetraVectorInput<tvector_t>(ctV));
    }
    catch (std::exception &e){
      TEST_FAIL_AND_EXIT(*comm, 0, 
        string("XpetraVectorInput ")+e.what(), 1);
    }
  
    if (rank==0)
      std::cout << "Input adapter for Tpetra::Vector" << std::endl;
    
    fail = verifyInputAdapter<tvector_t>(*tVInput, *tV);
  
    gfail = globalFail(comm, fail);
  
    if (!gfail){
      Array<lno_t> partitionNum(vlen,0);  // Migrate all elements to proc 0
      tvector_t *vMigrate = NULL;
      try{
        tVInput->applyPartitioningSolution(*tV, vMigrate,
          vlen, nprocs, vtxGids.getRawPtr(), vtxLids,
          partitionNum.getRawPtr());
        newV = rcp(vMigrate);
      }
      catch (std::exception &e){
        fail = 11;
      }

      gfail = globalFail(comm, fail);
  
      if (!gfail){
        RCP<const tvector_t> cnewV = rcp_const_cast<const tvector_t>(newV);
        RCP<Zoltan2::XpetraVectorInput<tvector_t> > newInput;
        try{
          newInput = rcp(new Zoltan2::XpetraVectorInput<tvector_t>(cnewV));
        }
        catch (std::exception &e){
          TEST_FAIL_AND_EXIT(*comm, 0, 
            string("XpetraVectorInput 2 ")+e.what(), 1);
        }
  
        if (rank==0){
          std::cout << 
           "Input adapter for Tpetra::Vector migrated to proc 0" << 
           std::endl;
        }
        fail = verifyInputAdapter<tvector_t>(*newInput, *newV);
        if (fail) fail += 100;
        gfail = globalFail(comm, fail);
      }
    }
    if (gfail){
      printFailureCode(comm, fail);
    }
  }

  /////////////////////////////////////////////////////////////
  // User object is Xpetra::Vector
  if (!gfail){ 
    RCP<xvector_t> xV = uinput->getXpetraVector();
    RCP<const xvector_t> cxV = rcp_const_cast<const xvector_t>(xV);
    RCP<Zoltan2::XpetraVectorInput<xvector_t> > xVInput;
  
    try {
      xVInput = 
        rcp(new Zoltan2::XpetraVectorInput<xvector_t>(cxV));
    }
    catch (std::exception &e){
      TEST_FAIL_AND_EXIT(*comm, 0, 
        string("XpetraVectorInput 3 ")+e.what(), 1);
    }
  
    if (rank==0){
      std::cout << "Input adapter for Xpetra::Vector" << std::endl;
    }
    fail = verifyInputAdapter<xvector_t>(*xVInput, *tV);
  
    gfail = globalFail(comm, fail);
  
    if (!gfail){
      Array<lno_t> partitionNum(vlen,0);  // Migrate all elements to proc 0
      xvector_t *vMigrate =NULL;
       try{
        xVInput->applyPartitioningSolution(*xV, vMigrate, 
          vlen, nprocs, vtxGids.getRawPtr(), vtxLids,
          partitionNum.getRawPtr());
      }
      catch (std::exception &e){
        fail = 11;
      }
  
      gfail = globalFail(comm, fail);
  
      if (!gfail){
        RCP<const xvector_t> cnewV(vMigrate);
        RCP<Zoltan2::XpetraVectorInput<xvector_t> > newInput;
        try{
          newInput = 
            rcp(new Zoltan2::XpetraVectorInput<xvector_t>(cnewV));
        }
        catch (std::exception &e){
          TEST_FAIL_AND_EXIT(*comm, 0, 
            string("XpetraVectorInput 4 ")+e.what(), 1);
        }
  
        if (rank==0){
          std::cout << 
           "Input adapter for Xpetra::Vector migrated to proc 0" << 
           std::endl;
        }
        fail = verifyInputAdapter<xvector_t>(*newInput, *newV);
        if (fail) fail += 100;
        gfail = globalFail(comm, fail);
      }
    }
    if (gfail){
      printFailureCode(comm, fail);
    }
  }

  /////////////////////////////////////////////////////////////
  // User object is Epetra_Vector
  if (!gfail){ 
    RCP<evector_t> eV = uinput->getEpetraVector();
    RCP<const evector_t> ceV = rcp_const_cast<const evector_t>(eV);
    RCP<Zoltan2::XpetraVectorInput<evector_t> > eVInput;
  
    try {
      eVInput = 
        rcp(new Zoltan2::XpetraVectorInput<evector_t>(ceV));
    }
    catch (std::exception &e){
      TEST_FAIL_AND_EXIT(*comm, 0, 
        string("XpetraVectorInput 5 ")+e.what(), 1);
    }
  
    if (rank==0){
      std::cout << "Input adapter for Epetra_Vector" << std::endl;
    }
    fail = verifyInputAdapter<evector_t>(*eVInput, *tV);
  
    gfail = globalFail(comm, fail);
  
    if (!gfail){
      Array<lno_t> partitionNum(vlen,0);  // Migrate all elements to proc 0
      evector_t *vMigrate =NULL;
      try{
        eVInput->applyPartitioningSolution(*eV, vMigrate,
          vlen, nprocs, vtxGids.getRawPtr(), vtxLids,
          partitionNum.getRawPtr());
      }
      catch (std::exception &e){
        fail = 11;
      }
  
      gfail = globalFail(comm, fail);
  
      if (!gfail){
        RCP<const evector_t> cnewV(vMigrate, true);
        RCP<Zoltan2::XpetraVectorInput<evector_t> > newInput;
        try{
          newInput = 
            rcp(new Zoltan2::XpetraVectorInput<evector_t>(cnewV));
        }
        catch (std::exception &e){
          TEST_FAIL_AND_EXIT(*comm, 0, 
            string("XpetraVectorInput 6 ")+e.what(), 1);
        }
  
        if (rank==0){
          std::cout << 
           "Input adapter for Epetra_Vector migrated to proc 0" << 
           std::endl;
        }
        fail = verifyInputAdapter<evector_t>(*newInput, *newV);
        if (fail) fail += 100;
        gfail = globalFail(comm, fail);
      }
    }
    if (gfail){
      printFailureCode(comm, fail);
    }
  }

  /////////////////////////////////////////////////////////////
  // DONE

  if (rank==0)
    std::cout << "PASS" << std::endl;
}
