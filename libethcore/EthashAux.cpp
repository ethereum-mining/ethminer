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

#define ETH_IGNORE_EXCEPTIONS(X) try { X; } catch (...) {}

EthashAux* dev::eth::EthashAux::s_this = nullptr;

EthashAux::~EthashAux()
{
	while (!m_lights.empty())
		killCache(m_lights.begin()->first);
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

ethash_params EthashAux::params(h256 const& _seedHash)
{
	RecursiveGuard l(get()->x_this);
	unsigned epoch = 0;
	try
	{
		epoch = get()->m_seedHashes.at(_seedHash);
	}
	catch (...)
	{
		for (h256 h; h != _seedHash && epoch < 2048; ++epoch, h = h256(h)) {}
		if (epoch == 2048)
		{
			std::ostringstream error;
			error << "apparent block number for " << _seedHash.abridged() << " is too high; max is " << (ETHASH_EPOCH_LENGTH * 2048);
			throw std::invalid_argument(error.str());
		}
		get()->m_seedHashes[_seedHash] = epoch;
	}
	return params(epoch * ETHASH_EPOCH_LENGTH);
}

void EthashAux::killCache(h256 const& _s)
{
	RecursiveGuard l(x_this);
	if (m_lights.count(_s))
	{
		ethash_delete_light(m_lights.at(_s));
		m_lights.erase(_s);
	}
}

void const* EthashAux::light(BlockInfo const& _header)
{
	return light(_header.seedHash());
}

void const* EthashAux::light(h256 const& _seedHash)
{
	RecursiveGuard l(get()->x_this);
	if (!get()->m_lights.count(_seedHash))
	{
		ethash_params p = params(_seedHash);
		get()->m_lights[_seedHash] = ethash_new_light(&p, _seedHash.data());
	}
	return get()->m_lights[_seedHash];
}

bytesConstRef EthashAux::full(BlockInfo const& _header, bytesRef _dest)
{
	return full(_header.seedHash(), _dest);
}

bytesConstRef EthashAux::full(h256 const& _seedHash, bytesRef _dest)
{
	RecursiveGuard l(get()->x_this);
	if (get()->m_fulls.count(_seedHash) && _dest)
	{
		assert(get()->m_fulls.size() <= _dest.size());
		get()->m_fulls.at(_seedHash).copyTo(_dest);
		return _dest;
	}
	if (!get()->m_fulls.count(_seedHash))
	{
		// @memoryleak @bug place it on a pile for deletion - perhaps use shared_ptr.
/*		if (!m_fulls.empty())
		{
			delete [] m_fulls.begin()->second.data();
			m_fulls.erase(m_fulls.begin());
		}*/

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

		ETH_IGNORE_EXCEPTIONS(boost::filesystem::remove(oldMemoFile));
		ETH_IGNORE_EXCEPTIONS(boost::filesystem::remove(oldMemoFile + ".info"));

		ethash_params p = params(_seedHash);
		assert(!_dest || _dest.size() >= p.full_size);	// must be big enough.

		bytesRef r = contentsNew(memoFile, _dest);
		if (!r)
		{
			// file didn't exist.
			if (_dest)
				// buffer was passed in - no insertion into cache nor need to allocate
				r = _dest;
			else
				r = bytesRef(new byte[p.full_size], p.full_size);
			ethash_prep_full(r.data(), &p, light(_seedHash));
			writeFile(memoFile, r);
		}
		if (_dest)
			return _dest;
		get()->m_fulls[_seedHash] = r;
	}
	return get()->m_fulls[_seedHash];
}

Ethash::Result EthashAux::eval(BlockInfo const& _header, Nonce const& _nonce)
{
	return eval(_header.seedHash(), _header.headerHash(WithoutNonce), _nonce);
}

Ethash::Result EthashAux::eval(h256 const& _seedHash, h256 const& _headerHash, Nonce const& _nonce)
{
	auto p = EthashAux::params(_seedHash);
	ethash_return_value r;
	if (EthashAux::get()->m_fulls.count(_seedHash))
		ethash_compute_full(&r, EthashAux::get()->full(_seedHash).data(), &p, _headerHash.data(), (uint64_t)(u64)_nonce);
	else
		ethash_compute_light(&r, EthashAux::get()->light(_seedHash), &p, _headerHash.data(), (uint64_t)(u64)_nonce);
//	cdebug << "EthashAux::eval sha3(cache):" << sha3(EthashAux::get()->cache(_header)) << "hh:" << _header.headerHash(WithoutNonce) << "nonce:" << _nonce << " => " << h256(r.result, h256::ConstructFromPointer);
	return Ethash::Result{h256(r.result, h256::ConstructFromPointer), h256(r.mix_hash, h256::ConstructFromPointer)};
}
