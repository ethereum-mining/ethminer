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
/** @file CanonBlockChain.h
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 */

#pragma once

#pragma warning(push)
#pragma warning(disable: 4100 4267)
#include <leveldb/db.h>
#pragma warning(pop)

#include <mutex>
#include <libdevcore/Log.h>
#include <libdevcore/Exceptions.h>
#include <libethcore/Common.h>
#include <libethcore/BlockInfo.h>
#include <libdevcore/Guards.h>
#include "BlockDetails.h"
#include "Account.h"
#include "BlockChain.h"
namespace ldb = leveldb;

namespace dev
{

namespace eth
{

// TODO: Move all this Genesis stuff into Genesis.h/.cpp
std::unordered_map<Address, Account> const& genesisState();

/**
 * @brief Implements the blockchain database. All data this gives is disk-backed.
 * @threadsafe
 * @todo Make not memory hog (should actually act as a cache and deallocate old entries).
 */
class CanonBlockChain: public BlockChain
{
public:
		CanonBlockChain(WithExisting _we = WithExisting::Trust, ProgressCallback const& _pc = ProgressCallback()): CanonBlockChain(std::string(), _we, _pc) {}
		CanonBlockChain(std::string const& _path, WithExisting _we = WithExisting::Trust, ProgressCallback const& _pc = ProgressCallback());
		~CanonBlockChain() {}

		/// @returns the genesis block header.
		static BlockInfo const& genesis() { UpgradableGuard l(x_genesis); if (!s_genesis) { auto gb = createGenesisBlock(); UpgradeGuard ul(l); s_genesis.reset(new BlockInfo); s_genesis->populate(&gb); } return *s_genesis; }

		/// @returns the genesis block as its RLP-encoded byte array.
		/// @note This is slow as it's constructed anew each call. Consider genesis() instead.
		static bytes createGenesisBlock();

private:
		/// Static genesis info and its lock.
		static boost::shared_mutex x_genesis;
		static std::unique_ptr<BlockInfo> s_genesis;
};

}
}
