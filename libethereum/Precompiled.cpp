/*
	This file is part of cpp-ethereum.

	cpp-ethereum is free software: you can redistribute it and/or modify
	it under the terms of the GNU General Public License as published by
	the Free Software Foundation, either version 3 of the License, or
	(at your option) any later version.

	cpp-ethereum is distributed in the hope that it will be useful,
	but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	GNU General Public License for more details.

	You should have received a copy of the GNU General Public License
	along with cpp-ethereum.  If not, see <http://www.gnu.org/licenses/>.
*/
/** @file Precompiled.cpp
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 */

#include "Precompiled.h"

#include <libdevcrypto/SHA3.h>
#include <libdevcrypto/Common.h>
#include <libethcore/CommonEth.h>
using namespace std;
using namespace dev;
using namespace dev::eth;

static bytes ecrecoverCode(bytesConstRef _in)
{
	struct inType
	{
		h256 hash;
		h256 v;
		h256 r;
		h256 s;
	} in;

	memcpy(&in, _in.data(), min(_in.size(), sizeof(in)));

	h256 ret;

	if ((u256)in.v > 28)
		return ret.asBytes();
	SignatureStruct sig(in.r, in.s, (byte)((int)(u256)in.v - 27));
	if (!sig.isValid())
		return ret.asBytes();

	try
	{
		ret = dev::sha3(recover(sig, in.hash));
	}
	catch (...) {}

	memset(ret.data(), 0, 12);
	return ret.asBytes();
}

static bytes sha256Code(bytesConstRef _in)
{
	bytes ret(32);
	sha256(_in, &ret);
	return ret;
}

static bytes ripemd160Code(bytesConstRef _in)
{
	bytes ret(32);
	ripemd160(_in, &ret);
	// leaves the 20-byte hash left-aligned. we want it right-aligned:
	memmove(ret.data() + 12, ret.data(), 20);
	memset(ret.data(), 0, 12);
	return ret;
}

static bytes identityCode(bytesConstRef _in)
{
	return _in.toBytes();
}

static const std::map<unsigned, PrecompiledAddress> c_precompiled =
{
	{ 1, { [](bytesConstRef) -> bigint { return (bigint)500; }, ecrecoverCode }},
	{ 2, { [](bytesConstRef i) -> bigint { return (bigint)50 + (i.size() + 31) / 32 * 50; }, sha256Code }},
	{ 3, { [](bytesConstRef i) -> bigint { return (bigint)50 + (i.size() + 31) / 32 * 50; }, ripemd160Code }},
	{ 4, { [](bytesConstRef i) -> bigint { return (bigint)1 + (i.size() + 31) / 32 * 1; }, identityCode }}
};

std::map<unsigned, PrecompiledAddress> const& dev::eth::precompiled()
{
	return c_precompiled;
}
