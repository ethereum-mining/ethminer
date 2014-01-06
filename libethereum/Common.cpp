/*
	This file is part of cpp-ethereum.

	cpp-ethereum is free software: you can redistribute it and/or modify
	it under the terms of the GNU General Public License as published by
	the Free Software Foundation, either version 3 of the License, or
	(at your option) any later version.

	Foobar is distributed in the hope that it will be useful,
	but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	GNU General Public License for more details.

	You should have received a copy of the GNU General Public License
	along with Foobar.  If not, see <http://www.gnu.org/licenses/>.
*/
/** @file Common.cpp
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 */

#include <random>
#include "Common.h"
#include "Exceptions.h"
#include "rmd160.h"
using namespace std;
using namespace eth;

std::string eth::escaped(std::string const& _s, bool _all)
{
	std::string ret;
	ret.reserve(_s.size());
	ret.push_back('"');
	for (auto i: _s)
		if (i == '"' && !_all)
			ret += "\\\"";
		else if (i == '\\' && !_all)
			ret += "\\\\";
		else if (i < ' ' || i > 127 || _all)
		{
			ret += "\\x";
			ret.push_back("0123456789abcdef"[(uint8_t)i / 16]);
			ret.push_back("0123456789abcdef"[(uint8_t)i % 16]);
		}
		else
			ret.push_back(i);
	ret.push_back('"');
	return ret;
}

std::string eth::randomWord()
{
	static std::mt19937_64 s_eng(0);
	std::string ret(std::uniform_int_distribution<int>(4, 10)(s_eng), ' ');
	char const n[] = "qwertyuiopasdfghjklzxcvbnmQWERTYUIOPASDFGHJKLZXCVBNM1234567890";
	std::uniform_int_distribution<int> d(0, sizeof(n) - 2);
	for (char& c: ret)
		c = n[d(s_eng)];
	return ret;
}

int eth::fromHex(char _i)
{
	if (_i >= '0' && _i <= '9')
		return _i - '0';
	if (_i >= 'a' && _i <= 'f')
		return _i - 'a' + 10;
	if (_i >= 'A' && _i <= 'F')
		return _i - 'A' + 10;
	throw BadHexCharacter();
}

bytes eth::fromUserHex(std::string const& _s)
{
	assert(_s.size() % 2 == 0);
	if (_s.size() < 2)
		return bytes();
	uint s = (_s[0] == '0' && _s[1] == 'x') ? 2 : 0;
	std::vector<uint8_t> ret;
	ret.reserve((_s.size() - s) / 2);
	for (uint i = s; i < _s.size(); i += 2)
		ret.push_back(fromHex(_s[i]) * 16 + fromHex(_s[i + 1]));
	return ret;
}

bytes eth::toHex(std::string const& _s)
{
	std::vector<uint8_t> ret;
	ret.reserve(_s.size() * 2);
	for (auto i: _s)
	{
		ret.push_back(i / 16);
		ret.push_back(i % 16);
	}
	return ret;
}

// /////////////////////////////////////////////////
// RIPEMD-160 stuff. Leave well alone.
// /////////////////////////////////////////////////

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
