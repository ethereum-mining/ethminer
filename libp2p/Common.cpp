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
/** @file Common.cpp
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 */

#include "Common.h"

#include <libethential/RLP.h>
#include "PeerSession.h"
#include "PeerHost.h"
using namespace std;
using namespace eth;
using namespace p2p;

// Helper function to determine if an address falls within one of the reserved ranges
// For V4:
// Class A "10.*", Class B "172.[16->31].*", Class C "192.168.*"
// Not implemented yet for V6
bool p2p::isPrivateAddress(bi::address _addressToCheck)
{
	if (_addressToCheck.is_v4())
	{
		bi::address_v4 v4Address = _addressToCheck.to_v4();
		bi::address_v4::bytes_type bytesToCheck = v4Address.to_bytes();
		if (bytesToCheck[0] == 10 || bytesToCheck[0] == 127)
			return true;
		if (bytesToCheck[0] == 172 && (bytesToCheck[1] >= 16 && bytesToCheck[1] <= 31))
			return true;
		if (bytesToCheck[0] == 192 && bytesToCheck[1] == 168)
			return true;
	}
	return false;
}

std::string p2p::reasonOf(DisconnectReason _r)
{
	switch (_r)
	{
	case DisconnectRequested: return "Disconnect was requested.";
	case TCPError: return "Low-level TCP communication error.";
	case BadProtocol: return "Data format error.";
	case UselessPeer: return "Peer had no use for this node.";
	case TooManyPeers: return "Peer had too many connections.";
	case DuplicatePeer: return "Peer was already connected.";
	case IncompatibleProtocol: return "Peer protocol versions are incompatible.";
	case ClientQuit: return "Peer is exiting.";
	default: return "Unknown reason.";
	}
}

void PeerCapability::disable(std::string const& _problem)
{
	clog(NetConnect) << "Disabling capability '" << m_host->name() << "'. Reason:" << _problem;
	m_enabled = false;
}

void HostCapabilityFace::seal(bytes& _b)
{
	m_host->seal(_b);
}

std::vector<std::shared_ptr<PeerSession> > HostCapabilityFace::peers() const
{
	Guard l(m_host->x_peers);
	std::vector<std::shared_ptr<PeerSession> > ret;
	for (auto const& i: m_host->m_peers)
		if (std::shared_ptr<PeerSession> p = i.second.lock())
			if (p->m_capabilities.count(name()))
				ret.push_back(p);
	return ret;
}

RLPStream& PeerCapability::prep(RLPStream& _s)
{
	return PeerSession::prep(_s);
}

void PeerCapability::sealAndSend(RLPStream& _s)
{
	m_session->sealAndSend(_s);
}

void PeerCapability::sendDestroy(bytes& _msg)
{
	m_session->sendDestroy(_msg);
}

void PeerCapability::send(bytesConstRef _msg)
{
	m_session->send(_msg);
}

void PeerCapability::addRating(unsigned _r)
{
	m_session->addRating(_r);
}
