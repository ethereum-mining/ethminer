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
using namespace std;
using namespace dev;
using namespace dev::p2p;

const unsigned dev::p2p::c_protocolVersion = 4;
const unsigned dev::p2p::c_defaultIPPort = 30303;
static_assert(dev::p2p::c_protocolVersion == 4, "Replace v3 compatbility with v4 compatibility before updating network version.");

const dev::p2p::NodeIPEndpoint dev::p2p::UnspecifiedNodeIPEndpoint = NodeIPEndpoint(bi::address(), 0, 0);
const dev::p2p::Node dev::p2p::UnspecifiedNode = dev::p2p::Node(NodeId(), UnspecifiedNodeIPEndpoint);

bool dev::p2p::NodeIPEndpoint::test_allowLocal = false;

//⊳⊲◀▶■▣▢□▷◁▧▨▩▲◆◉◈◇◎●◍◌○◼☑☒☎☢☣☰☀♽♥♠✩✭❓✔✓✖✕✘✓✔✅⚒⚡⦸⬌∅⁕«««»»»⚙━┅┉▬

#ifdef _WIN32
const char* NetWarn::name() { return EthYellow "N" EthRed " X"; }
const char* NetImpolite::name() { return EthYellow "N" EthRed " !"; }
const char* NetNote::name() { return EthYellow "N" EthBlue " i"; }
const char* NetConnect::name() { return EthYellow "N" EthYellow " C"; }
const char* NetMessageSummary::name() { return EthYellow "N" EthWhite " ."; }
const char* NetMessageDetail::name() { return EthYellow "N" EthGray " o"; }
const char* NetTriviaSummary::name() { return EthYellow "N" EthGray " O"; }
const char* NetTriviaDetail::name() { return EthYellow "N" EthCoal " 0"; }
const char* NetAllDetail::name() { return EthYellow "N" EthCoal " A"; }
const char* NetRight::name() { return EthYellow "N" EthGreen "->"; }
const char* NetLeft::name() { return EthYellow "N" EthNavy "<-"; }
const char* NetP2PWarn::name() { return EthYellow "N" EthRed " X"; }
const char* NetP2PNote::name() { return EthYellow "N" EthBlue " i"; }
const char* NetP2PConnect::name() { return EthYellow "N" EthYellow " C"; }
#else
const char* NetWarn::name() { return EthYellow "⧎" EthRed " ✘"; }
const char* NetImpolite::name() { return EthYellow "⧎" EthRed " !"; }
const char* NetNote::name() { return EthYellow "⧎" EthBlue " ℹ"; }
const char* NetConnect::name() { return EthYellow "⧎" EthYellow " ▢"; }
const char* NetMessageSummary::name() { return EthYellow "⧎" EthWhite " ◌"; }
const char* NetMessageDetail::name() { return EthYellow "⧎" EthGray " ○"; }
const char* NetTriviaSummary::name() { return EthYellow "⧎" EthGray " ◎"; }
const char* NetTriviaDetail::name() { return EthYellow "⧎" EthCoal " ◍"; }
const char* NetAllDetail::name() { return EthYellow "⧎" EthCoal " ●"; }
const char* NetRight::name() { return EthYellow "⧎" EthGreen "▬▶"; }
const char* NetLeft::name() { return EthYellow "⧎" EthNavy "◀▬"; }
const char* NetP2PWarn::name() { return EthYellow "⧎" EthRed " ✘"; }
const char* NetP2PNote::name() { return EthYellow "⧎" EthBlue " ℹ"; }
const char* NetP2PConnect::name() { return EthYellow "⧎" EthYellow " ▢"; }
#endif

bool p2p::isPublicAddress(std::string const& _addressToCheck)
{
	return _addressToCheck.empty() ? false : isPublicAddress(bi::address::from_string(_addressToCheck));
}

bool p2p::isPublicAddress(bi::address const& _addressToCheck)
{
	return !(isPrivateAddress(_addressToCheck) || isLocalHostAddress(_addressToCheck));
}

// Helper function to determine if an address falls within one of the reserved ranges
// For V4:
// Class A "10.*", Class B "172.[16->31].*", Class C "192.168.*"
bool p2p::isPrivateAddress(bi::address const& _addressToCheck)
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
	else if (_addressToCheck.is_v6())
	{
		bi::address_v6 v6Address = _addressToCheck.to_v6();
		bi::address_v6::bytes_type bytesToCheck = v6Address.to_bytes();
		if (bytesToCheck[0] == 0xfd && bytesToCheck[1] == 0)
			return true;
		if (!bytesToCheck[0] && !bytesToCheck[1] && !bytesToCheck[2] && !bytesToCheck[3] && !bytesToCheck[4] && !bytesToCheck[5] && !bytesToCheck[6] && !bytesToCheck[7]
				 && !bytesToCheck[8] && !bytesToCheck[9] && !bytesToCheck[10] && !bytesToCheck[11] && !bytesToCheck[12] && !bytesToCheck[13] && !bytesToCheck[14] && (bytesToCheck[15] == 0 || bytesToCheck[15] == 1))
			return true;
	}
	return false;
}

bool p2p::isPrivateAddress(std::string const& _addressToCheck)
{
	return _addressToCheck.empty() ? false : isPrivateAddress(bi::address::from_string(_addressToCheck));
}

// Helper function to determine if an address is localhost
bool p2p::isLocalHostAddress(bi::address const& _addressToCheck)
{
	// @todo: ivp6 link-local adresses (macos), ex: fe80::1%lo0
	static const set<bi::address> c_rejectAddresses = {
		{bi::address_v4::from_string("127.0.0.1")},
		{bi::address_v4::from_string("0.0.0.0")},
		{bi::address_v6::from_string("::1")},
		{bi::address_v6::from_string("::")}
	};
	
	return find(c_rejectAddresses.begin(), c_rejectAddresses.end(), _addressToCheck) != c_rejectAddresses.end();
}

bool p2p::isLocalHostAddress(std::string const& _addressToCheck)
{
	return _addressToCheck.empty() ? false : isLocalHostAddress(bi::address::from_string(_addressToCheck));
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
	case NullIdentity: return "Null identity given.";
	case ClientQuit: return "Peer is exiting.";
	case UnexpectedIdentity: return "Unexpected identity given.";
	case LocalIdentity: return "Connected to ourselves.";
	case UserReason: return "Subprotocol reason.";
	case NoDisconnect: return "(No disconnect has happened.)";
	default: return "Unknown reason.";
	}
}

void NodeIPEndpoint::streamRLP(RLPStream& _s, RLPAppend _append) const
{
	if (_append == StreamList)
		_s.appendList(3);
	if (address.is_v4())
		_s << bytesConstRef(&address.to_v4().to_bytes()[0], 4);
	else if (address.is_v6())
		_s << bytesConstRef(&address.to_v6().to_bytes()[0], 16);
	else
		_s << bytes();
	_s << udpPort << tcpPort;
}

void NodeIPEndpoint::interpretRLP(RLP const& _r)
{
	if (_r[0].size() == 4)
		address = bi::address_v4(*(bi::address_v4::bytes_type*)_r[0].toBytes().data());
	else if (_r[0].size() == 16)
		address = bi::address_v6(*(bi::address_v6::bytes_type*)_r[0].toBytes().data());
	else
		address = bi::address();
	udpPort = _r[1].toInt<uint16_t>();
	tcpPort = _r[2].toInt<uint16_t>();
}

void DeadlineOps::reap()
{
	if (m_stopped)
		return;

	Guard l(x_timers);
	std::vector<DeadlineOp>::iterator t = m_timers.begin();
	while (t != m_timers.end())
		if (t->expired())
		{
			t->wait();
			t = m_timers.erase(t);
		}
		else
			t++;

	m_timers.emplace_back(m_io, m_reapIntervalMs, [this](boost::system::error_code const& ec)
	{
		if (!ec && !m_stopped)
			reap();
	});
}

namespace dev {
	
std::ostream& operator<<(std::ostream& _out, dev::p2p::NodeIPEndpoint const& _ep)
{
	_out << _ep.address << _ep.udpPort << _ep.tcpPort;
	return _out;
}
	
}

