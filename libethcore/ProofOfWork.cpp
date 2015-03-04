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
/** @file ProofOfWork.cpp
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 */

#include <boost/detail/endian.hpp>
#include <boost/filesystem.hpp>
#include <chrono>
#include <array>
#include <random>
#include <thread>
#include <libdevcore/Guards.h>
#include <libdevcore/Log.h>
#include <libdevcrypto/CryptoPP.h>
#include <libdevcrypto/FileSystem.h>
#include <libdevcore/Common.h>
#include <libethash/ethash.h>
#include "BlockInfo.h"
#include "ProofOfWork.h"
using namespace std;
using namespace std::chrono;

namespace dev
{
namespace eth
{

class Ethasher
{
public:
	Ethasher() {}

	static Ethasher* get() { if (!s_this) s_this = new Ethasher(); return s_this; }

	bytes const& cache(BlockInfo const& _header)
	{
		RecursiveGuard l(x_this);
		if (!m_caches.count(_header.seedHash))
		{
			try {
				boost::filesystem::create_directories(getDataDir() + "/ethashcache");
			} catch (...) {}
			std::string memoFile = getDataDir() + "/ethashcache/" + toHex(_header.seedHash.ref().cropped(0, 4)) + ".cache";
			m_caches[_header.seedHash] = contents(memoFile);
			if (m_caches[_header.seedHash].empty())
			{
				ethash_params p = params((unsigned)_header.number);
				m_caches[_header.seedHash].resize(p.cache_size);
				ethash_prep_light(m_caches[_header.seedHash].data(), &p, _header.seedHash.data());
				writeFile(memoFile, m_caches[_header.seedHash]);
			}
		}
		cdebug << "sha3 of cache: " << sha3(m_caches[_header.seedHash]);
		return m_caches[_header.seedHash];
	}

	byte const* full(BlockInfo const& _header)
	{
		RecursiveGuard l(x_this);
		if (!m_fulls.count(_header.seedHash))
		{
			if (!m_fulls.empty())
			{
				delete [] m_fulls.begin()->second.data();
				m_fulls.erase(m_fulls.begin());
			}
			std::string memoFile = getDataDir() + "/ethashcache/" + toHex(_header.seedHash.ref().cropped(0, 4)) + ".full";
			m_fulls[_header.seedHash] = contentsNew(memoFile);
			if (!m_fulls[_header.seedHash])
			{
				ethash_params p = params((unsigned)_header.number);
				m_fulls[_header.seedHash] = bytesRef(new byte[p.full_size], p.full_size);
				auto c = cache(_header);
				ethash_prep_full(m_fulls[_header.seedHash].data(), &p, c.data());
				writeFile(memoFile, m_fulls[_header.seedHash]);
			}
		}
		cdebug << "sha3 of full pad: " << sha3(m_fulls[_header.seedHash]);
		return m_fulls[_header.seedHash].data();
	}

	static ethash_params params(BlockInfo const& _header)
	{
		return params((unsigned)_header.number);
	}

	static ethash_params params(unsigned _n)
	{
		ethash_params p;
		p.cache_size = ethash_get_cachesize(_n);
		p.full_size = ethash_get_datasize(_n);
		return p;
	}

private:
	static Ethasher* s_this;
	RecursiveMutex x_this;
	std::map<h256, bytes> m_caches;
	std::map<h256, bytesRef> m_fulls;
};

Ethasher* Ethasher::s_this = nullptr;

bool Ethash::verify(BlockInfo const& _header)
{
	bigint boundary = (bigint(1) << 256) / _header.difficulty;
	auto e = eval(_header, _header.nonce);
	return (u256)e.value <= boundary && e.mixHash == _header.mixHash;
}

Ethash::Result Ethash::eval(BlockInfo const& _header, Nonce const& _nonce)
{
	auto p = Ethasher::params(_header);
	ethash_return_value r;
	ethash_compute_light(&r, Ethasher::get()->cache(_header).data(), &p, _header.headerHash(WithoutNonce).data(), (uint64_t)(u64)_nonce);
	return Result{h256(r.result, h256::ConstructFromPointer), h256(r.mix_hash, h256::ConstructFromPointer)};
}

std::pair<MineInfo, Ethash::Proof> Ethash::mine(BlockInfo const& _header, unsigned _msTimeout, bool _continue, bool _turbo)
{
	auto h = _header.headerHash(WithoutNonce);
	auto p = Ethasher::params(_header);
	auto d = Ethasher::get()->full(_header);

	std::pair<MineInfo, Proof> ret;
	static std::mt19937_64 s_eng((time(0) + *reinterpret_cast<unsigned*>(m_last.data())));
	uint64_t tryNonce = (uint64_t)(u64)(m_last = Nonce::random(s_eng));

	bigint boundary = (bigint(1) << 256) / _header.difficulty;
	ret.first.requirement = log2((double)boundary);

	// 2^ 0      32      64      128      256
	//   [--------*-------------------------]
	//
	// evaluate until we run out of time
	auto startTime = std::chrono::steady_clock::now();
	if (!_turbo)
		std::this_thread::sleep_for(std::chrono::milliseconds(_msTimeout * 90 / 100));
	double best = 1e99;	// high enough to be effectively infinity :)
	Proof result;
	ethash_return_value ethashReturn;
	unsigned hashCount = 0;
	for (; (std::chrono::steady_clock::now() - startTime) < std::chrono::milliseconds(_msTimeout) && _continue; tryNonce++, hashCount++)
	{
		ethash_compute_full(&ethashReturn, d, &p, h.data(), tryNonce);
		u256 val(h256(ethashReturn.result, h256::ConstructFromPointer));
		best = std::min<double>(best, log2((double)val));
		if (val <= boundary)
		{
			ret.first.completed = true;
			result.mixHash = h256(ethashReturn.mix_hash, h256::ConstructFromPointer);
			result.nonce = u64(tryNonce);
			break;
		}
	}
	ret.first.hashes = hashCount;
	ret.first.best = best;
	ret.second = result;

	if (ret.first.completed)
	{
		BlockInfo test = _header;
		assignResult(result, test);
		assert(verify(test));
	}

	return ret;
}

}
}
