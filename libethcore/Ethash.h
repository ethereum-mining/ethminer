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

namespace dev
{

class RLP;
class RLPStream;

namespace eth
{

class BlockInfo;
class EthashCLHook;

class Ethash
{
public:
	static std::string name();
	static unsigned revision();
	static SealEngineFace* createSealEngine();

	// TODO: remove or virtualize
	struct Solution
	{
		h64 nonce;
		h256 mixHash;
	};
	// TODO: make private
	struct Result
	{
		h256 value;
		h256 mixHash;
	};
	// TODO: virtualise
	struct WorkPackage
	{
		WorkPackage() = default;

		void reset() { headerHash = h256(); }
		operator bool() const { return headerHash != h256(); }

		h256 boundary;
		h256 headerHash;	///< When h256() means "pause until notified a new work package is available".
		h256 seedHash;
	};
	static const WorkPackage NullWorkPackage;

	class BlockHeaderRaw: public BlockInfo
	{
		friend class EthashSeal;

	public:
		bool verify() const;
		bool preVerify() const;

		void prep(std::function<int(unsigned)> const& _f = std::function<int(unsigned)>()) const;
		WorkPackage package() const;
		h256 const& seedHash() const;
		h64 const& nonce() const { return m_nonce; }
		h256 const& mixHash() const { return m_mixHash; }

	protected:
		BlockHeaderRaw(BlockInfo const& _bi): BlockInfo(_bi) {}

		static const unsigned SealFields = 2;

		void populateFromHeader(RLP const& _header, Strictness _s);
		void clear() { m_mixHash = h256(); m_nonce = h64(); }
		void noteDirty() const { m_seedHash = h256(); }
		void streamRLPFields(RLPStream& _s) const { _s << m_mixHash << m_nonce; }

	private:
		h64 m_nonce;
		h256 m_mixHash;

		mutable h256 m_seedHash;
		mutable h256 m_hash;						///< SHA3 hash of the block header! Not serialised.
	};
	using BlockHeader = BlockHeaderPolished<BlockHeaderRaw>;

	// TODO: Move elsewhere (EthashAux?)
	static void ensurePrecomputed(unsigned _number);

	/// Default value of the local work size. Also known as workgroup size.
	static const unsigned defaultLocalWorkSize;
	/// Default value of the global work size as a multiplier of the local work size
	static const unsigned defaultGlobalWorkSizeMultiplier;
	/// Default value of the milliseconds per global work size (per batch)
	static const unsigned defaultMSPerBatch;
};

}
}
