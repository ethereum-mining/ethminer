/********************************************************************\
 *
 *      FILE:     rmd160.h
 *
 *      CONTENTS: Header file for a sample C-implementation of the
 *                RIPEMD-160 hash-function.
 *      TARGET:   any computer with an ANSI C compiler
 *
 *      AUTHOR:   Antoon Bosselaers, ESAT-COSIC
 *      DATE:     1 March 1996
 *      VERSION:  1.0
 *
 *      Copyright (c) Katholieke Universiteit Leuven
 *      1996, All Rights Reserved
 *
\********************************************************************/

#ifndef  RMD160H           /* make sure this file is read only once */
#define  RMD160H

#include <cstdint>

/********************************************************************/

/* function prototypes */

void MDinit(uint32_t *MDbuf);
/*
 *  initializes MDbuffer to "magic constants"
 */

void compress(uint32_t *MDbuf, uint32_t *X);
/*
 *  the compression function.
 *  transforms MDbuf using message bytes X[0] through X[15]
 */

void MDfinish(uint32_t *MDbuf, unsigned char const *strptr, uint32_t lswlen, uint32_t mswlen);
/*
 *  puts bytes from strptr into X and pad out; appends length
 *  and finally, compresses the last block(s)
 *  note: length in bits == 8 * (lswlen + 2^32 mswlen).
 *  note: there are (lswlen mod 64) bytes left in strptr.
 */

#endif  /* RMD160H */

/*********************** end of file rmd160.h ***********************/
