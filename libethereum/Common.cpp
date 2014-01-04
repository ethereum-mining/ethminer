#include "Common.h"
#include "rmd160.h"
using namespace std;
using namespace eth;

/* collect four bytes into one word: */
#define BYTES_TO_DWORD(strptr)                    \
			(((uint32_t) *((strptr)+3) << 24) | \
			 ((uint32_t) *((strptr)+2) << 16) | \
			 ((uint32_t) *((strptr)+1) <<  8) | \
			 ((uint32_t) *(strptr)))

u256 eth::ripemd160(bytesConstRef _message)
/*
 * returns RMD(message)
 * message should be a string terminated by '\0'
 */
{
	static const uint RMDsize = 160;
   uint32_t         MDbuf[RMDsize/32];   /* contains (A, B, C, D(, E))   */
   static byte   hashcode[RMDsize/8]; /* for final hash-value         */
   uint32_t         X[16];               /* current 16-word chunk        */
   unsigned int  i;                   /* counter                      */
   uint32_t         length;              /* length in bytes of message   */
   uint32_t         nbytes;              /* # of bytes not yet processed */

   /* initialize */
   MDinit(MDbuf);
   length = _message.size();
   auto message = _message.data();

   /* process message in 16-word chunks */
   for (nbytes=length; nbytes > 63; nbytes-=64) {
	  for (i=0; i<16; i++) {
		 X[i] = BYTES_TO_DWORD(message);
		 message += 4;
	  }
	  compress(MDbuf, X);
   }                                    /* length mod 64 bytes left */

   /* finish: */
   MDfinish(MDbuf, message, length, 0);

   for (i=0; i<RMDsize/8; i+=4) {
	  hashcode[i]   =  MDbuf[i>>2];         /* implicit cast to byte  */
	  hashcode[i+1] = (MDbuf[i>>2] >>  8);  /*  extracts the 8 least  */
	  hashcode[i+2] = (MDbuf[i>>2] >> 16);  /*  significant bits.     */
	  hashcode[i+3] = (MDbuf[i>>2] >> 24);
   }

   u256 ret = 0;
   for (i = 0; i < RMDsize / 8; ++i)
	   ret = (ret << 8) | hashcode[i];
   return ret;
}
