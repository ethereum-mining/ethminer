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
/** @file Ethash.h
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 *
 * A proof of work algorithm.
 */

#pragma once

#include <chrono>
#include <thread>
#include <cstdint>
#include <libdevcore/CommonIO.h>
#include "Common.h"
#include "Miner.h"
#include "Farm.h"
#include "Sealer.h"

class ethash_cl_miner;
class ethash_cuda_miner;

namespace dev
{

class RLP;
class RLPStream;

namespace eth
{

class BlockInfo;
class EthashCLHook;
class EthashCUDAHook;

class Ethash
{
public:
	static std::string name();
	static unsigned revision();
	static SealEngineFace* createSealEngine();

	using Nonce = h64;

	static void manuallySubmitWork(SealEngineFace* _engine, h256 const& _mixHash, Nonce _nonce);
	static bool isWorking(SealEngineFace* _engine);
	static WorkingProgress workingProgress(SealEngineFace* _engine);

	class BlockHeaderRaw: public BlockInfo
	{
		friend class EthashSealEngine;

	public:
		static const unsigned SealFields = 2;

		bool verify() const;
		bool preVerify() const;

		void prep(std::function<int(unsigned)> const& _f = std::function<int(unsigned)>()) const;
		h256 const& seedHash() const;
		Nonce const& nonce() const { return m_nonce; }
		h256 const& mixHash() const { return m_mixHash; }

		void setNonce(Nonce const& _n) { m_nonce = _n; noteDirty(); }
		void setMixHash(h256 const& _n) { m_mixHash = _n; noteDirty(); }

		StringHashMap jsInfo() const;

	protected:
		BlockHeaderRaw() = default;
		BlockHeaderRaw(BlockInfo const& _bi): BlockInfo(_bi) {}

		void populateFromHeader(RLP const& _header, Strictness _s);
		void populateFromParent(BlockHeaderRaw const& _parent);
		void verifyParent(BlockHeaderRaw const& _parent);
		void clear() { m_mixHash = h256(); m_nonce = Nonce(); }
		void noteDirty() const { m_seedHash = h256(); }
		void streamRLPFields(RLPStream& _s) const { _s << m_mixHash << m_nonce; }

	private:
		Nonce m_nonce;
		h256 m_mixHash;

		mutable h256 m_seedHash;
		mutable h256 m_hash;						///< SHA3 hash of the block header! Not serialised.
	};
	using BlockHeader = BlockHeaderPolished<BlockHeaderRaw>;

	static void manuallySetWork(SealEngineFace* _engine, BlockHeader const& _work);

	// TODO: Move elsewhere (EthashAux?)
	static void ensurePrecomputed(unsigned _number);
};

}
}
