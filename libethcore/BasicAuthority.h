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
/** @file BasicAuthority.h
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 *
 * Determines the PoW algorithm.
 */

#pragma once

#include <libdevcore/RLP.h>
#include <libdevcrypto/Common.h>
#include "BlockInfo.h"
#include "Common.h"
#include "Sealer.h"

class BasicAuthoritySeal;
class BasicAuthoritySealEngine;

namespace dev
{
namespace eth
{

/**
 * The proof of work algorithm base type.
 *
 * Must implement a basic templated interface, including:
 * typename Result
 * typename Solution
 * typename CPUMiner
 * typename GPUMiner
 * and a few others. TODO
 */
class BasicAuthority
{
	friend class ::BasicAuthoritySealEngine;

public:
	static std::string name() { return "BasicAuthority"; }
	static unsigned revision() { return 0; }
	static SealEngineFace* createSealEngine();

	class BlockHeaderRaw: public BlockInfo
	{
		friend class ::BasicAuthoritySealEngine;

	public:
		static const unsigned SealFields = 1;

		bool verify() const;
		bool preVerify() const;

		Signature sig() const { return m_sig; }

		StringHashMap jsInfo() const;

	protected:
		BlockHeaderRaw() = default;
		BlockHeaderRaw(BlockInfo const& _bi): BlockInfo(_bi) {}

		void populateFromHeader(RLP const& _header, Strictness _s);
		void populateFromParent(BlockHeaderRaw const& _parent);
		void verifyParent(BlockHeaderRaw const& _parent);
		void streamRLPFields(RLPStream& _s) const { _s << m_sig; }
		void clear() { m_sig = Signature(); }
		void noteDirty() const {}

	private:
		Signature m_sig;
	};
	using BlockHeader = BlockHeaderPolished<BlockHeaderRaw>;

private:
	static AddressHash s_authorities;
};

}
}
