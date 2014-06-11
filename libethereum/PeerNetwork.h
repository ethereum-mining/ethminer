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
 * Miscellanea required for the PeerServer/PeerSession classes.
 */

#pragma once

#include <string>
#include <boost/asio.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <chrono>
#include <libethsupport/Common.h>
#include <libethsupport/Log.h>
namespace ba = boost::asio;
namespace bi = boost::asio::ip;

namespace eth
{

bool isPrivateAddress(bi::address _addressToCheck);

class OverlayDB;
class BlockChain;
class TransactionQueue;
class PeerServer;
class PeerSession;

struct NetWarn: public LogChannel { static const char* name() { return "!N!"; } static const int verbosity = 0; };
struct NetNote: public LogChannel { static const char* name() { return "*N*"; } static const int verbosity = 1; };
struct NetMessageSummary: public LogChannel { static const char* name() { return "-N-"; } static const int verbosity = 2; };
struct NetMessageDetail: public LogChannel { static const char* name() { return "=N="; } static const int verbosity = 3; };
struct NetTriviaSummary: public LogChannel { static const char* name() { return "-N-"; } static const int verbosity = 4; };
struct NetTriviaDetail: public LogChannel { static const char* name() { return "=N="; } static const int verbosity = 5; };
struct NetAllDetail: public LogChannel { static const char* name() { return "=N="; } static const int verbosity = 6; };
struct NetRight: public LogChannel { static const char* name() { return ">N>"; } static const int verbosity = 8; };
struct NetLeft: public LogChannel { static const char* name() { return "<N<"; } static const int verbosity = 9; };

enum PacketType
{
	HelloPacket = 0,
	DisconnectPacket,
	PingPacket,
	PongPacket,
	GetPeersPacket = 0x10,
	PeersPacket,
	TransactionsPacket,
	BlocksPacket,
	GetChainPacket,
	NotInChainPacket,
	GetTransactionsPacket
};

enum DisconnectReason
{
	DisconnectRequested = 0,
	TCPError,
	BadProtocol,
	UselessPeer,
	TooManyPeers,
	DuplicatePeer,
	WrongGenesis,
	IncompatibleProtocol,
	ClientQuit
};

/// @returns the string form of the given disconnection reason.
std::string reasonOf(DisconnectReason _r);

struct PeerInfo
{
	std::string clientVersion;
	std::string host;
	unsigned short port;
	std::chrono::steady_clock::duration lastPing;
};

class UPnP;

enum class NodeMode
{
	Full,
	PeerServer
};

}
