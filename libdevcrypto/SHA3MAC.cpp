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
/** @file SHA3MAC.cpp
 * @author Alex Leverington <nessence@gmail.com>
 * @date 2014
 *
 * SHA3 MAC
 */

#pragma GCC diagnostic ignored "-Wunused-function"
#pragma warning(push)
#pragma warning(disable:4100 4244)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wconversion"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wunused-variable"
#pragma GCC diagnostic ignored "-Wdelete-non-virtual-dtor"
#pragma GCC diagnostic ignored "-Wextra"
#include <sha3.h>
#pragma warning(pop)
#pragma GCC diagnostic pop
#include "SHA3MAC.h"

using namespace dev;
using namespace dev::crypto;
using namespace CryptoPP;

void sha3mac(bytesConstRef _secret, bytesConstRef _plain, bytesRef _output)
{
	CryptoPP::SHA3_256 ctx;
	ctx.Update((byte*)_secret.data(), _secret.size());
	ctx.Update((byte*)_plain.data(), _plain.size());
	assert(_output.size() >= 32);
	ctx.Final(_output.data());
}

