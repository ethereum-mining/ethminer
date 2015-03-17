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
/** @file Ethasher.h
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 *
 * ProofOfWork algorithm.
 */

#pragma once

#include <chrono>
#include <thread>
#include <cstdint>
#include <libdevcore/Guards.h>
#include <libdevcrypto/SHA3.h>
#include <libethash/ethash.h>		// TODO: REMOVE once everything merged into this class and an opaque API can be provided.
#define ETHASH_REVISION REVISION
#define ETHASH_DATASET_BYTES_INIT DATASET_BYTES_INIT
#define ETHASH_DATASET_BYTES_GROWTH DATASET_BYTES_GROWTH
#define ETHASH_CACHE_BYTES_INIT CACHE_BYTES_INIT
#define ETHASH_CACHE_BYTES_GROWTH CACHE_BYTES_GROWTH
#define ETHASH_DAGSIZE_BYTES_INIT DAGSIZE_BYTES_INIT
#define ETHASH_DAG_GROWTH DAG_GROWTH
#define ETHASH_EPOCH_LENGTH EPOCH_LENGTH
#define ETHASH_MIX_BYTES MIX_BYTES
#define ETHASH_HASH_BYTES HASH_BYTES
#define ETHASH_DATASET_PARENTS DATASET_PARENTS
#define ETHASH_CACHE_ROUNDS CACHE_ROUNDS
#define ETHASH_ACCESSES ACCESSES
#undef REVISION
#undef DATASET_BYTES_INIT
#undef DATASET_BYTES_GROWTH
#undef CACHE_BYTES_INIT
#undef CACHE_BYTES_GROWTH
#undef DAGSIZE_BYTES_INIT
#undef DAG_GROWTH
#undef EPOCH_LENGTH
#undef MIX_BYTES
#undef HASH_BYTES
#undef DATASET_PARENTS
#undef CACHE_ROUNDS
#undef ACCESSES

#include "Common.h"
#include "BlockInfo.h"

namespace dev
{
namespace eth
{

class Ethasher
{
public:
	Ethasher() {}

	static Ethasher* get() { if (!s_this) s_this = new Ethasher(); return s_this; }

	bytes const& cache(BlockInfo const& _header);
	bytesConstRef full(BlockInfo const& _header);
	static ethash_params params(BlockInfo const& _header);
	static ethash_params params(unsigned _n);

	struct Result
	{
		h256 value;
		h256 mixHash;
	};

	static Result eval(BlockInfo const& _header) { return eval(_header, _header.nonce); }
	static Result eval(BlockInfo const& _header, Nonce const& _nonce);
	static bool verify(BlockInfo const& _header);

	class Miner
	{
	public:
		Miner(BlockInfo const& _header):
			m_headerHash(_header.headerHash(WithoutNonce)),
			m_params(Ethasher::params(_header)),
			m_datasetPointer(Ethasher::get()->full(_header).data())
		{}

		inline h256 mine(uint64_t _nonce)
		{
			ethash_compute_full(&m_ethashReturn, m_datasetPointer, &m_params, m_headerHash.data(), _nonce);
			return h256(m_ethashReturn.result, h256::ConstructFromPointer);
		}

		inline h256 lastMixHash() const
		{
			return h256(m_ethashReturn.mix_hash, h256::ConstructFromPointer);
		}

	private:
		ethash_return_value m_ethashReturn;
		h256 m_headerHash;
		ethash_params m_params;
		void const* m_datasetPointer;
	};

private:
	static Ethasher* s_this;
	RecursiveMutex x_this;
	std::map<h256, bytes> m_caches;
	std::map<h256, bytesRef> m_fulls;
};

}
}
