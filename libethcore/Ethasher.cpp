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
/** @file Ethasher.cpp
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 */

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
#include "Ethasher.h"
using namespace std;
using namespace chrono;
using namespace dev;
using namespace eth;

Ethasher* dev::eth::Ethasher::s_this = nullptr;

bytes const& Ethasher::cache(BlockInfo const& _header)
{
	RecursiveGuard l(x_this);
	if (_header.number > c_ethashEpochLength * 2048)
	{
		std::ostringstream error;
		error << "block number is too high; max is " << c_ethashEpochLength * 2048 << "(was " << _header.number << ")";
		throw std::invalid_argument( error.str() );
 	}

	if (!m_caches.count(_header.seedHash()))
	{
		ethash_params p = params((unsigned)_header.number);
		m_caches[_header.seedHash()].resize(p.cache_size);
		ethash_prep_light(m_caches[_header.seedHash()].data(), &p, _header.seedHash().data());
	}
	return m_caches[_header.seedHash()];
}

bytesConstRef Ethasher::full(BlockInfo const& _header)
{
	RecursiveGuard l(x_this);
	if (!m_fulls.count(_header.seedHash()))
	{
		if (!m_fulls.empty())
		{
			delete [] m_fulls.begin()->second.data();
			m_fulls.erase(m_fulls.begin());
		}
		try {
			boost::filesystem::create_directories(getDataDir("ethash"));
		} catch (...) {}

		std::string memoFile = getDataDir("ethash") + "/full";
		auto info = rlpList(c_ethashRevision, _header.seedHash());
		if (boost::filesystem::exists(memoFile) && contents(memoFile + ".info") != info)
			boost::filesystem::remove(memoFile);
		m_fulls[_header.seedHash()] = contentsNew(memoFile);
		if (!m_fulls[_header.seedHash()])
		{
			ethash_params p = params((unsigned)_header.number);
			m_fulls[_header.seedHash()] = bytesRef(new byte[p.full_size], p.full_size);
			auto c = cache(_header);
			ethash_prep_full(m_fulls[_header.seedHash()].data(), &p, c.data());
			writeFile(memoFile, m_fulls[_header.seedHash()]);
			writeFile(memoFile + ".info", info);
		}
	}
	return m_fulls[_header.seedHash()];
}

ethash_params Ethasher::params(BlockInfo const& _header)
{
	return params((unsigned)_header.number);
}

ethash_params Ethasher::params(unsigned _n)
{
	ethash_params p;
	p.cache_size = ethash_get_cachesize(_n);
	p.full_size = ethash_get_datasize(_n);
	return p;
}

bool Ethasher::verify(BlockInfo const& _header)
{
	if (_header.number >= c_ethashEpochLength * 2048)
		return false;

	h256 boundary = u256((bigint(1) << 256) / _header.difficulty);

	bool quick = ethash_quick_check_difficulty(
		_header.headerHash(WithoutNonce).data(),
		(uint64_t)(u64)_header.nonce,
		_header.mixHash.data(),
		boundary.data());

#if !ETH_DEBUG
	if (!quick)
		return false;
#endif

	auto result = eval(_header);
	bool slow = result.value <= boundary && result.mixHash == _header.mixHash;

#if ETH_DEBUG
	if (!quick && slow)
	{
		cwarn << "WARNING: evaluated result gives true whereas ethash_quick_check_difficulty gives false.";
		cwarn << "headerHash:" << _header.headerHash(WithoutNonce);
		cwarn << "nonce:" << _header.nonce;
		cwarn << "mixHash:" << _header.mixHash;
		cwarn << "difficulty:" << _header.difficulty;
		cwarn << "boundary:" << boundary;
		cwarn << "result.value:" << result.value;
		cwarn << "result.mixHash:" << result.mixHash;
	}
#endif

	return slow;
}

Ethasher::Result Ethasher::eval(BlockInfo const& _header, Nonce const& _nonce)
{
	auto p = Ethasher::params(_header);
	ethash_return_value r;
	ethash_compute_light(&r, Ethasher::get()->cache(_header).data(), &p, _header.headerHash(WithoutNonce).data(), (uint64_t)(u64)_nonce);
//	cdebug << "Ethasher::eval sha3(cache):" << sha3(Ethasher::get()->cache(_header)) << "hh:" << _header.headerHash(WithoutNonce) << "nonce:" << _nonce << " => " << h256(r.result, h256::ConstructFromPointer);
	return Result{h256(r.result, h256::ConstructFromPointer), h256(r.mix_hash, h256::ConstructFromPointer)};
}
