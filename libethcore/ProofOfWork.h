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
/** @file ProofOfWork.h
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 *
 * Determines the PoW algorithm.
 */

#pragma once

#include <libdevcore/RLP.h>
#include <libdevcrypto/Common.h>
#include "Common.h"
//#include "Ethash.h"

namespace dev
{
namespace eth
{

class BlockInfo;

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
public:
	struct HeaderCache {};
	static void ensureHeaderCacheValid(HeaderCache&, BlockInfo const&) {}
	static void composeException(Exception&, BlockInfo&) {}
	static void composeExceptionPre(Exception&, BlockInfo&) {}

	struct Solution
	{
		bool operator==(Solution const& _v) const { return sig == _v.sig; }
		void populateFromRLP(RLP const& io_rlp, int& io_field)
		{
			sig = io_rlp[io_field++].toHash<dev::Signature>(RLP::VeryStrict);
		}

		void streamRLP(RLPStream& io_rlp) const
		{
			io_rlp << sig;
		}
		static const unsigned Fields = 1;
		Signature sig;
	};

	struct Result
	{
		Signature sig;
	};

	struct WorkPackage
	{
		void reset() { headerHash = h256(); }
		operator bool() const { return headerHash != h256(); }

		h256 headerHash;	///< When h256() means "pause until notified a new work package is available".
	};

	static const WorkPackage NullWorkPackage;

	static std::string name() { return "BasicAuthority"; }
	static unsigned revision() { return 0; }
	static void prep(BlockInfo const&, std::function<int(unsigned)> const& = std::function<int(unsigned)>()) {}
	static void ensurePrecomputed(unsigned) {}
	static bool verify(BlockInfo const& _header);
	static bool preVerify(BlockInfo const& _header);
	static WorkPackage package(BlockInfo const& _header);

	static const Address Authority;

	struct Farm
	{
	public:
		strings sealers() const { return { "default" }; }
		void setSealer(std::string const&) {}
		void setSecret(Secret const& _s) { m_secret = _s; }
		void sealBlock(BlockInfo const& _bi);
		void disable() {}
		void onSolutionFound(std::function<void(Solution const& s)> const& _f) { m_onSolutionFound = _f; }
		bool isMining() const { return false; }
		MiningProgress miningProgress() const { return MiningProgress(); }

	private:
		Secret m_secret;
		std::function<void(Solution const& s)> m_onSolutionFound;
	};
};

using ProofOfWork = BasicAuthority;

}
}
