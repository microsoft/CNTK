// Constants.h -- the constants used by CNTK
//

#pragma once

#ifndef _CONSTANTS_H_
#define _CONSTANTS_H_

// Constants used in SGD distributed gradient aggregation.
// The default threshold size to pack a gradient into a continuous buffer during aggregation for less MPI ops.
const size_t DEFAULT_PACK_THRESHOLD_SIZE_IN_KB = 32;
const size_t DEFAULT_PACK_THRESHOLD_SIZE_IN_BYTES = DEFAULT_PACK_THRESHOLD_SIZE_IN_KB * 1024;

#endif
