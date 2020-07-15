#ifndef __Zoltan2_color_util_h__
#define __Zoltan2_color_util_h__

#include <stdlib.h>
#include <sys/time.h>

double timer() {
  struct timeval tp;
  gettimeofday(&tp, NULL);
  return ((double) (tp.tv_sec) + 1e-6 * tp.tv_usec);
}

#endif
