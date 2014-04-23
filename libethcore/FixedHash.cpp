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
/** @file FixedHash.cpp
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 */

#include "FixedHash.h"
#include "CryptoHeaders.h"

using namespace std;
using namespace eth;

h256 eth::EmptySHA3 = sha3(bytesConstRef());

std::string eth::sha3(std::string const& _input, bool _hex)
{
	if (!_hex)
	{
		string ret(32, '\0');
		sha3(bytesConstRef((byte const*)_input.data(), _input.size()), bytesRef((byte*)ret.data(), 32));
		return ret;
	}

	uint8_t buf[32];
	sha3(bytesConstRef((byte const*)_input.data(), _input.size()), bytesRef((byte*)&(buf[0]), 32));
	std::string ret(64, '\0');
	for (unsigned int i = 0; i < 32; i++)
		sprintf((char*)(ret.data())+i*2, "%02x", buf[i]);
	return ret;
}

void eth::sha3(bytesConstRef _input, bytesRef _output)
{
	CryptoPP::SHA3_256 ctx;
	ctx.Update((byte*)_input.data(), _input.size());
	assert(_output.size() >= 32);
	ctx.Final(_output.data());
}

bytes eth::sha3Bytes(bytesConstRef _input)
{
	bytes ret(32);
	sha3(_input, &ret);
	return ret;
}

h256 eth::sha3(bytesConstRef _input)
{
	h256 ret;
	sha3(_input, bytesRef(&ret[0], 32));
	return ret;
}
