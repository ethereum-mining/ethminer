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

#include <libdevcore/Log.h>
#include <libdevcore/SHA3.h>
#include <libdevcore/Hash.h>
#include <libdevcrypto/Common.h>
#include <libethcore/Common.h>
#include <libevmcore/Params.h>
using namespace std;
using namespace dev;
using namespace dev::eth;

namespace
{

void ecrecoverCode(bytesConstRef _in, bytesRef _out)
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
	u256 v = (u256)in.v;
	if (v >= 27 && v <= 28)
	{
		SignatureStruct sig(in.r, in.s, (byte)((int)v - 27));
		if (sig.isValid())
		{
			try
			{
				Public rec = recover(sig, in.hash);
				if (rec)
					ret = dev::sha3(rec);
				else
					return;
			}
			catch (...) { return; }
		}
	}

	memset(ret.data(), 0, 12);
	ret.ref().copyTo(_out);
}

void sha256Code(bytesConstRef _in, bytesRef _out)
{
	sha256(_in).ref().copyTo(_out);
}

void ripemd160Code(bytesConstRef _in, bytesRef _out)
{
	h256(ripemd160(_in), h256::AlignRight).ref().copyTo(_out);
}

void identityCode(bytesConstRef _in, bytesRef _out)
{
	_in.copyTo(_out);
}

}

std::unordered_map<unsigned, PrecompiledAddress> const& dev::eth::precompiled()
{
	static const std::unordered_map<unsigned, PrecompiledAddress> c_precompiled =
	{
		{ 1, { [](bytesConstRef) -> bigint { return c_ecrecoverGas; }, ecrecoverCode }},
		{ 2, { [](bytesConstRef i) -> bigint { return c_sha256Gas + (i.size() + 31) / 32 * c_sha256WordGas; }, sha256Code }},
		{ 3, { [](bytesConstRef i) -> bigint { return c_ripemd160Gas + (i.size() + 31) / 32 * c_ripemd160WordGas; }, ripemd160Code }},
		{ 4, { [](bytesConstRef i) -> bigint { return c_identityGas + (i.size() + 31) / 32 * c_identityWordGas; }, identityCode }}
	};
	return c_precompiled;
}
