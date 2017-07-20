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
/** @file EthashAux.cpp
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 */

#pragma once

#include <condition_variable>
#include <libethash/ethash.h>
#include <libdevcore/Log.h>
#include <libdevcore/Worker.h>
#include "BlockHeader.h"

namespace dev
{
namespace eth
{

struct Solution
{
	uint64_t nonce;
	h256 mixHash;
	h256 headerHash;
	h256 seedHash;
	h256 boundary;
};

struct Result
{
	h256 value;
	h256 mixHash;
};

class EthashAux
{
public:
	struct LightAllocation
	{
		LightAllocation(h256 const& _seedHash);
		~LightAllocation();
		bytesConstRef data() const;
		Result compute(h256 const& _headerHash, uint64_t _nonce) const;
		ethash_light_t light;
		uint64_t size;
	};

	using LightType = std::shared_ptr<LightAllocation>;

	static h256 seedHash(unsigned _number);
	static uint64_t number(h256 const& _seedHash);

	static LightType light(h256 const& _seedHash);

	static Result eval(h256 const& _seedHash, h256 const& _headerHash, uint64_t  _nonce) noexcept;

private:
	EthashAux() = default;
	static EthashAux& get();

	Mutex x_lights;
	std::unordered_map<h256, LightType> m_lights;

	Mutex x_epochs;
	std::unordered_map<h256, unsigned> m_epochs;
	h256s m_seedHashes;
};

struct WorkPackage
{
	WorkPackage() = default;
	WorkPackage(BlockHeader const& _bh) :
		boundary(_bh.boundary()),
		header(_bh.hashWithout()),
		seed(EthashAux::seedHash(static_cast<unsigned>(_bh.number())))
	{ }
	void reset() { header = h256(); }
	operator bool() const { return header != h256(); }

	h256 boundary;
	h256 header;	///< When h256() means "pause until notified a new work package is available".
	h256 seed;

	uint64_t startNonce = 0;
	int exSizeBits = -1;
};

}
}
