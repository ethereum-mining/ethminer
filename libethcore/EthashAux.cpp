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

#include "EthashAux.h"

#include <boost/detail/endian.hpp>
#include <boost/filesystem.hpp>
#include <chrono>
#include <array>
#include <random>
#include <thread>
#include <libdevcore/Common.h>
#include <libdevcore/Guards.h>
#include <libdevcore/Log.h>
#include <libdevcrypto/CryptoPP.h>
#include <libdevcrypto/SHA3.h>
#include <libdevcrypto/FileSystem.h>
#include <libethash/internal.h>
#include "BlockInfo.h"
#include "Exceptions.h"
using namespace std;
using namespace chrono;
using namespace dev;
using namespace eth;

EthashAux* dev::eth::EthashAux::s_this = nullptr;

EthashAux::~EthashAux()
{
}

uint64_t EthashAux::cacheSize(BlockInfo const& _header)
{
	return ethash_get_cachesize((uint64_t)_header.number);
}

h256 EthashAux::seedHash(unsigned _number)
{
	unsigned epoch = _number / ETHASH_EPOCH_LENGTH;
	Guard l(get()->x_epochs);
	if (epoch >= get()->m_seedHashes.size())
	{
		h256 ret;
		unsigned n = 0;
		if (!get()->m_seedHashes.empty())
		{
			ret = get()->m_seedHashes.back();
			n = get()->m_seedHashes.size() - 1;
		}
		get()->m_seedHashes.resize(epoch + 1);
//		cdebug << "Searching for seedHash of epoch " << epoch;
		for (; n <= epoch; ++n, ret = sha3(ret))
		{
			get()->m_seedHashes[n] = ret;
//			cdebug << "Epoch" << n << "is" << ret;
		}
	}
	return get()->m_seedHashes[epoch];
}

void EthashAux::killCache(h256 const& _s)
{
	RecursiveGuard l(x_this);
	m_lights.erase(_s);
}

EthashAux::LightType EthashAux::light(BlockInfo const& _header)
{
	return light((uint64_t)_header.number);
}

EthashAux::LightType EthashAux::light(uint64_t _blockNumber)
{
	RecursiveGuard l(get()->x_this);
	h256 seedHash = EthashAux::seedHash(_blockNumber);
	LightType ret = get()->m_lights[seedHash];
	return ret ? ret : (get()->m_lights[seedHash] = make_shared<LightAllocation>(_blockNumber));
}

EthashAux::LightAllocation::LightAllocation(uint64_t _blockNumber)
{
	light = ethash_light_new(_blockNumber);
	size = ethash_get_cachesize(_blockNumber);
}

EthashAux::LightAllocation::~LightAllocation()
{
	ethash_light_delete(light);
}

bytesConstRef EthashAux::LightAllocation::data() const
{
	return bytesConstRef((byte const*)light->cache, size);
}

EthashAux::FullAllocation::FullAllocation(ethash_light_t _light, ethash_callback_t _cb)
{
	full = ethash_full_new(_light, _cb);
}

EthashAux::FullAllocation::~FullAllocation()
{
	ethash_full_delete(full);
}

bytesConstRef EthashAux::FullAllocation::data() const
{
	return bytesConstRef((byte const*)ethash_full_dag(full), size());
}

EthashAux::FullType EthashAux::full(BlockInfo const& _header)
{
	return full((uint64_t) _header.number);
}

EthashAux::FullType EthashAux::full(uint64_t _blockNumber)
{
	RecursiveGuard l(get()->x_this);
	h256 seedHash = EthashAux::seedHash(_blockNumber);
	FullType ret;
	if ((ret = get()->m_fulls[seedHash].lock()))
	{
		get()->m_lastUsedFull = ret;
		return ret;
	}
	ret = get()->m_lastUsedFull = make_shared<FullAllocation>(light(_blockNumber)->light, nullptr);
	get()->m_fulls[seedHash] = ret;
	return ret;
}

Ethash::Result EthashAux::FullAllocation::compute(h256 const& _headerHash, Nonce const& _nonce) const
{
	ethash_return_value_t r = ethash_full_compute(full, *(ethash_h256_t*)_headerHash.data(), (uint64_t)(u64)_nonce);
	if (!r.success)
		BOOST_THROW_EXCEPTION(DAGCreationFailure());
	return Ethash::Result{h256((uint8_t*)&r.result, h256::ConstructFromPointer), h256((uint8_t*)&r.mix_hash, h256::ConstructFromPointer)};
}

Ethash::Result EthashAux::LightAllocation::compute(h256 const& _headerHash, Nonce const& _nonce) const
{
	ethash_return_value r = ethash_light_compute(light, *(ethash_h256_t*)_headerHash.data(), (uint64_t)(u64)_nonce);
	if (!r.success)
		BOOST_THROW_EXCEPTION(DAGCreationFailure());
	return Ethash::Result{h256((uint8_t*)&r.result, h256::ConstructFromPointer), h256((uint8_t*)&r.mix_hash, h256::ConstructFromPointer)};
}

Ethash::Result EthashAux::eval(BlockInfo const& _header, Nonce const& _nonce)
{
	return eval((uint64_t)_header.number, _header.headerHash(WithoutNonce), _nonce);
}

Ethash::Result EthashAux::eval(uint64_t _blockNumber, h256 const& _headerHash, Nonce const& _nonce)
{
	h256 seedHash = EthashAux::seedHash(_blockNumber);
	if (FullType dag = get()->m_fulls[seedHash].lock())
		return dag->compute(_headerHash, _nonce);
	return EthashAux::get()->light(_blockNumber)->compute(_headerHash, _nonce);
}
