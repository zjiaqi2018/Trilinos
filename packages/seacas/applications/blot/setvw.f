C Copyright(C) 2009 Sandia Corporation. Under the terms of Contract
C DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
C certain rights in this software.
C         
C Redistribution and use in source and binary forms, with or without
C modification, are permitted provided that the following conditions are
C met:
C 
C     * Redistributions of source code must retain the above copyright
C       notice, this list of conditions and the following disclaimer.
C 
C     * Redistributions in binary form must reproduce the above
C       copyright notice, this list of conditions and the following
C       disclaimer in the documentation and/or other materials provided
C       with the distribution.
C     * Neither the name of Sandia Corporation nor the names of its
C       contributors may be used to endorse or promote products derived
C       from this software without specific prior written permission.
C 
C THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
C "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
C LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
C A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
C OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
C SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
C LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
C DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
C THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
C (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
C OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

C $Log: setvw.f,v $
C Revision 1.2  2009/03/25 12:36:47  gdsjaar
C Add copyright and license notice to all files.
C Permission to assert copyright has been granted; blot is now open source, BSD
C
C Revision 1.1  1994/04/07 20:11:49  gdsjaar
C Initial checkin of ACCESS/graphics/blotII2
C
c Revision 1.2  1990/12/14  08:57:18  gdsjaar
c Added RCS Id and Log to all files
c
C=======================================================================
      SUBROUTINE SETVW (IVIEW, *)
C=======================================================================

C   --*** SETVW *** (MESH) Set up viewport for view
C   --   Written by Amy Gilkey - revised 02/12/85
C   --
C   --SETVW sets up the the viewport for the specified view.
C   --
C   --Parameters:
C   --   IVIEW - IN - the view number
C   --   * - return statement the cancel function is active
C   --
C   --Common Variables:
C   --   Uses DVIEW, WVIEW of /LAYOUT/

      PARAMETER (KLFT=1, KRGT=2, KBOT=3, KTOP=4)

      COMMON /LAYOUD/ DVIEW(KTOP,4), WVIEW(KTOP,4)

      LOGICAL MPVIEW, MPORT2, LDUM
      LOGICAL GRABRT

      IF (GRABRT ()) RETURN 1

      LDUM = MPVIEW (DVIEW(KLFT,IVIEW), DVIEW(KRGT,IVIEW),
     &   DVIEW(KBOT,IVIEW), DVIEW(KTOP,IVIEW))
      LDUM = MPORT2 (WVIEW(KLFT,IVIEW), WVIEW(KRGT,IVIEW),
     &   WVIEW(KBOT,IVIEW), WVIEW(KTOP,IVIEW))

      RETURN
      END
