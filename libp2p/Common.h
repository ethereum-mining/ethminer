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
/** @file Common.h
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 *
 * Miscellanea required for the Host/Session classes.
 */

#pragma once

#include <string>
#include <boost/asio.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <chrono>
#include <libethential/Common.h>
#include <libethential/Log.h>
#include <libethential/FixedHash.h>
namespace ba = boost::asio;
namespace bi = boost::asio::ip;

namespace eth
{
class RLP;
class RLPStream;
}

namespace p2p
{

using eth::LogChannel;
using eth::bytes;
using eth::h256;
using eth::h512;
using eth::bytesConstRef;
using eth::RLP;
using eth::RLPStream;

bool isPrivateAddress(bi::address _addressToCheck);

class UPnP;
class Capability;
class Host;
class Session;

struct NetWarn: public LogChannel { static const char* name() { return "!N!"; } static const int verbosity = 0; };
struct NetNote: public LogChannel { static const char* name() { return "*N*"; } static const int verbosity = 1; };
struct NetMessageSummary: public LogChannel { static const char* name() { return "-N-"; } static const int verbosity = 2; };
struct NetConnect: public LogChannel { static const char* name() { return "+N+"; } static const int verbosity = 4; };
struct NetMessageDetail: public LogChannel { static const char* name() { return "=N="; } static const int verbosity = 5; };
struct NetTriviaSummary: public LogChannel { static const char* name() { return "-N-"; } static const int verbosity = 10; };
struct NetTriviaDetail: public LogChannel { static const char* name() { return "=N="; } static const int verbosity = 11; };
struct NetAllDetail: public LogChannel { static const char* name() { return "=N="; } static const int verbosity = 13; };
struct NetRight: public LogChannel { static const char* name() { return ">N>"; } static const int verbosity = 14; };
struct NetLeft: public LogChannel { static const char* name() { return "<N<"; } static const int verbosity = 15; };

enum PacketType
{
	HelloPacket = 0,
	DisconnectPacket,
	PingPacket,
	PongPacket,
	GetPeersPacket,
	PeersPacket,
	UserPacket = 0x10
};

enum DisconnectReason
{
	DisconnectRequested = 0,
	TCPError,
	BadProtocol,
	UselessPeer,
	TooManyPeers,
	DuplicatePeer,
	IncompatibleProtocol,
	InvalidIdentity,
	ClientQuit,
	UserReason = 0x10
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

}
