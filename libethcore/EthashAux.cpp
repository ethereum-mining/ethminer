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
#include <libethcore/Params.h>
#include "BlockInfo.h"
using namespace std;
using namespace chrono;
using namespace dev;
using namespace eth;

EthashAux* dev::eth::EthashAux::s_this = nullptr;

EthashAux::~EthashAux()
{
}

ethash_params EthashAux::params(BlockInfo const& _header)
{
	return params((unsigned)_header.number);
}

ethash_params EthashAux::params(unsigned _n)
{
	ethash_params p;
	p.cache_size = ethash_get_cachesize(_n);
	p.full_size = ethash_get_datasize(_n);
	return p;
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

ethash_params EthashAux::params(h256 const& _seedHash)
{
	Guard l(get()->x_epochs);
	unsigned epoch = 0;
	try
	{
		epoch = get()->m_epochs.at(_seedHash);
	}
	catch (...)
	{
//		cdebug << "Searching for seedHash " << _seedHash;
		for (h256 h; h != _seedHash && epoch < 2048; ++epoch, h = sha3(h), get()->m_epochs[h] = epoch) {}
		if (epoch == 2048)
		{
			std::ostringstream error;
			error << "apparent block number for " << _seedHash << " is too high; max is " << (ETHASH_EPOCH_LENGTH * 2048);
			throw std::invalid_argument(error.str());
		}
	}
	return params(epoch * ETHASH_EPOCH_LENGTH);
}

void EthashAux::killCache(h256 const& _s)
{
	RecursiveGuard l(x_this);
	m_lights.erase(_s);
}

EthashAux::LightType EthashAux::light(BlockInfo const& _header)
{
	return light(_header.seedHash());
}

EthashAux::LightType EthashAux::light(h256 const& _seedHash)
{
	RecursiveGuard l(get()->x_this);
	LightType ret = get()->m_lights[_seedHash];
	return ret ? ret : (get()->m_lights[_seedHash] = make_shared<LightAllocation>(_seedHash));
}

EthashAux::LightAllocation::LightAllocation(h256 const& _seed)
{
	auto p = params(_seed);
	size = p.cache_size;
	light = ethash_new_light(&p, _seed.data());
}

EthashAux::LightAllocation::~LightAllocation()
{
	ethash_delete_light(light);
}


EthashAux::FullType EthashAux::full(BlockInfo const& _header, bytesRef _dest, bool _createIfMissing)
{
	return full(_header.seedHash(), _dest, _createIfMissing);
}

EthashAux::FullType EthashAux::full(h256 const& _seedHash, bytesRef _dest, bool _createIfMissing)
{
	RecursiveGuard l(get()->x_this);
	FullType ret = get()->m_fulls[_seedHash].lock();
	if (ret && _dest)
	{
		assert(ret->data.size() <= _dest.size());
		ret->data.copyTo(_dest);
		return FullType();
	}
	if (!ret)
	{
		// drop our last used cache sine we're allocating another 1GB.
		get()->m_lastUsedFull.reset();

		try {
			boost::filesystem::create_directories(getDataDir("ethash"));
		} catch (...) {}

		auto info = rlpList(Ethash::revision(), _seedHash);
		std::string oldMemoFile = getDataDir("ethash") + "/full";
		std::string memoFile = getDataDir("ethash") + "/full-R" + toString(ETHASH_REVISION) + "-" + toHex(_seedHash.ref().cropped(0, 8));
		if (boost::filesystem::exists(oldMemoFile) && contents(oldMemoFile + ".info") == info)
		{
			// memofile valid - rename.
			boost::filesystem::rename(oldMemoFile, memoFile);
		}

		DEV_IGNORE_EXCEPTIONS(boost::filesystem::remove(oldMemoFile));
		DEV_IGNORE_EXCEPTIONS(boost::filesystem::remove(oldMemoFile + ".info"));

		ethash_params p = params(_seedHash);
		assert(!_dest || _dest.size() >= p.full_size);	// must be big enough.

		bytesRef r = contentsNew(memoFile, _dest);
		if (!r)
		{
			if (!_createIfMissing)
				return FullType();
			// file didn't exist.
			if (_dest)
				// buffer was passed in - no insertion into cache nor need to allocate
				r = _dest;
			else
				r = bytesRef(new byte[p.full_size], p.full_size);
			ethash_prep_full(r.data(), &p, light(_seedHash)->light);
			writeFile(memoFile, r);
		}
		if (_dest)
			return FullType();
		ret = make_shared<FullAllocation>(r);
		get()->m_fulls[_seedHash] = ret;
	}
	get()->m_lastUsedFull = ret;
	return ret;
}

Ethash::Result EthashAux::eval(BlockInfo const& _header, Nonce const& _nonce)
{
	return eval(_header.seedHash(), _header.headerHash(WithoutNonce), _nonce);
}

Ethash::Result EthashAux::FullAllocation::compute(h256 const& _seedHash, h256 const& _headerHash, Nonce const& _nonce) const
{
	ethash_return_value r;
	auto p = EthashAux::params(_seedHash);
	ethash_compute_full(&r, data.data(), &p, _headerHash.data(), (uint64_t)(u64)_nonce);
	return Ethash::Result{h256(r.result, h256::ConstructFromPointer), h256(r.mix_hash, h256::ConstructFromPointer)};
}

Ethash::Result EthashAux::LightAllocation::compute(h256 const& _seedHash, h256 const& _headerHash, Nonce const& _nonce) const
{
	ethash_return_value r;
	auto p = EthashAux::params(_seedHash);
	ethash_compute_light(&r, light, &p, _headerHash.data(), (uint64_t)(u64)_nonce);
	return Ethash::Result{h256(r.result, h256::ConstructFromPointer), h256(r.mix_hash, h256::ConstructFromPointer)};
}

Ethash::Result EthashAux::eval(h256 const& _seedHash, h256 const& _headerHash, Nonce const& _nonce)
{
	if (auto dag = EthashAux::get()->full(_seedHash, bytesRef(), false))
		return dag->compute(_seedHash, _headerHash, _nonce);
	return EthashAux::get()->light(_seedHash)->compute(_seedHash, _headerHash, _nonce);
}
