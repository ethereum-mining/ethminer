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
/** @file Message.h
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 */

#pragma once

#include <mutex>
#include <array>
#include <set>
#include <memory>
#include <utility>
#include <libdevcore/RLP.h>
#include <libdevcore/Guards.h>
#include <libdevcrypto/SHA3.h>
#include "Common.h"

namespace dev
{
namespace shh
{

struct Message
{
	unsigned expiry = 0;
	unsigned ttl = 0;
	bytes topic;
	bytes payload;

	Message() {}
	Message(unsigned _exp, unsigned _ttl, bytes const& _topic, bytes const& _payload): expiry(_exp), ttl(_ttl), topic(_topic), payload(_payload) {}
	Message(RLP const& _m)
	{
		expiry = _m[0].toInt<unsigned>();
		ttl = _m[1].toInt<unsigned>();
		topic = _m[2].toBytes();
		payload = _m[3].toBytes();
	}

	operator bool () const { return !!expiry; }

	void streamOut(RLPStream& _s) const { _s.appendList(4) << expiry << ttl << topic << payload; }
	h256 sha3() const { RLPStream s; streamOut(s); return dev::eth::sha3(s.out()); }
};

}
}
