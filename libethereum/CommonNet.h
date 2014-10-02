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
/** @file PeerNetwork.h
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 *
 * Miscellanea required for the PeerServer/Session classes.
 */

#pragma once

#include <string>
#include <chrono>
#include <libdevcore/Common.h>
#include <libdevcore/Log.h>

namespace dev
{
namespace eth
{

#if ETH_DEBUG
static const unsigned c_maxHashes = 64;		///< Maximum number of hashes BlockHashes will ever send.
static const unsigned c_maxHashesAsk = 64;		///< Maximum number of hashes GetBlockHashes will ever ask for.
static const unsigned c_maxBlocks = 32;		///< Maximum number of blocks Blocks will ever send.
static const unsigned c_maxBlocksAsk = 32;		///< Maximum number of blocks we ask to receive in Blocks (when using GetChain).
#else
static const unsigned c_maxHashes = 256;		///< Maximum number of hashes BlockHashes will ever send.
static const unsigned c_maxHashesAsk = 256;		///< Maximum number of hashes GetBlockHashes will ever ask for.
static const unsigned c_maxBlocks = 128;		///< Maximum number of blocks Blocks will ever send.
static const unsigned c_maxBlocksAsk = 128;		///< Maximum number of blocks we ask to receive in Blocks (when using GetChain).
#endif
class OverlayDB;
class BlockChain;
class TransactionQueue;
class EthereumHost;
class EthereumPeer;

enum EthereumPacket
{
	StatusPacket = 0x10,
	GetTransactionsPacket,
	TransactionsPacket,
	GetBlockHashesPacket,
	BlockHashesPacket,
	GetBlocksPacket,
	BlocksPacket,
};

enum class Grabbing
{
	State,
	Hashes,
	Chain,
	ChainHelper,
	Nothing
};

}
}
