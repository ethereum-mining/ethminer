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
/** @file Client.h
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 */

#pragma once

#include <thread>
#include <mutex>
#include <list>
#include <atomic>
#include <boost/utility.hpp>
#include <libdevcore/Common.h>
#include <libdevcore/CommonIO.h>
#include <libdevcore/Guards.h>
#include <libp2p/Host.h>

namespace dev
{

enum WorkState
{
	Active = 0,
	Deleting,
	Deleted
};

enum class NodeMode
{
	PeerServer,
	Full
};

namespace eth { class Interface; }
namespace shh { class Interface; }
namespace bzz { class Interface; }

/**
 * @brief Main API hub for interfacing with Web 3 components. This doesn't do any local multiplexing, so you can only have one
 * running on any given machine for the provided DB path.
 */
class RawWebThree
{
public:
	/// Constructor.
	RawWebThree(std::string const& _clientVersion, std::string const& _dbPath = std::string(), bool _forceClean = false);

	/// Destructor.
	~RawWebThree();

	// The mainline interfaces:

	eth::Interface* ethereum() const;
	shh::Interface* whisper() const;
	bzz::Interface* swarm() const;

	// Misc stuff:

	void setClientVersion(std::string const& _name) { m_clientVersion = _name; }

	// Network stuff:

	/// Get information on the current peer set.
	std::vector<p2p::PeerInfo> peers();
	/// Same as peers().size(), but more efficient.
	size_t peerCount() const;
	/// Same as peers().size(), but more efficient.
	void setIdealPeerCount(size_t _n) const;

	/// Start the network subsystem.
	void startNetwork(unsigned short _listenPort = 30303, std::string const& _remoteHost = std::string(), unsigned short _remotePort = 30303, NodeMode _mode = NodeMode::Full, unsigned _peers = 5, std::string const& _publicIP = std::string(), bool _upnp = true, dev::u256 _networkId = 0);
	/// Connect to a particular peer.
	void connect(std::string const& _seedHost, unsigned short _port = 30303);
	/// Stop the network subsystem.
	void stopNetwork();
	/// Is the network subsystem up?
	bool haveNetwork() { ReadGuard l(x_net); return !!m_net; }
	/// Save peers
	dev::bytes savePeers();
	/// Restore peers
	void restorePeers(bytesConstRef _saved);

private:
	/// Do some work on the network.
	void workNet();

	std::string m_clientVersion;				///< Our end-application client's name/version.

	std::unique_ptr<std::thread> m_workNet;		///< The network thread.
	std::atomic<WorkState> m_workNetState;
	mutable boost::shared_mutex x_net;			///< Lock for the network existance.
	std::unique_ptr<p2p::Host> m_net;			///< Should run in background and send us events when blocks found and allow us to send blocks as required.
};

}
