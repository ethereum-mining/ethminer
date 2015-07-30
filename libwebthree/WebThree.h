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
/** @file WebThree.h
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 */

#pragma once

#include <thread>
#include <mutex>
#include <list>
#include <atomic>
#include <boost/asio.hpp> // Make sure boost/asio.hpp is included before windows.h.
#include <boost/utility.hpp>
#include <libdevcore/Common.h>
#include <libdevcore/CommonIO.h>
#include <libdevcore/Guards.h>
#include <libdevcore/Exceptions.h>
#include <libp2p/Host.h>
#include <libwhisper/WhisperHost.h>
#include <libethereum/Client.h>

namespace dev
{

enum WorkState
{
	Active = 0,
	Deleting,
	Deleted
};

namespace eth { class Interface; }
namespace shh { class Interface; }
namespace bzz { class Interface; }


class WebThreeNetworkFace
{
public:
	/// Get information concerning this node.
	virtual p2p::NodeInfo nodeInfo() const = 0;

	/// Get information on the current peer set.
	virtual std::vector<p2p::PeerSessionInfo> peers() = 0;

	/// Same as peers().size(), but more efficient.
	virtual size_t peerCount() const = 0;

	/// Add node to connect to.
	virtual void addNode(p2p::NodeId const& _node, bi::tcp::endpoint const& _hostEndpoint) = 0;
	
	/// Require connection to peer.
	virtual void requirePeer(p2p::NodeId const& _node, bi::tcp::endpoint const& _endpoint) = 0;
	
	/// Save peers
	virtual dev::bytes saveNetwork() = 0;

	/// Sets the ideal number of peers.
	virtual void setIdealPeerCount(size_t _n) = 0;

	virtual bool haveNetwork() const = 0;

	virtual p2p::NetworkPreferences const& networkPreferences() const = 0;
	virtual void setNetworkPreferences(p2p::NetworkPreferences const& _n, bool _dropPeers) = 0;

	virtual p2p::NodeId id() const = 0;

	/// Gets the nodes.
	virtual p2p::Peers nodes() const = 0;

	/// Start the network subsystem.
	virtual void startNetwork() = 0;

	/// Stop the network subsystem.
	virtual void stopNetwork() = 0;

	/// Is network working? there may not be any peers yet.
	virtual bool isNetworkStarted() const = 0;

	/// Get enode string.
	virtual std::string enode() const = 0;
};


/**
 * @brief Main API hub for interfacing with Web 3 components. This doesn't do any local multiplexing, so you can only have one
 * running on any given machine for the provided DB path.
 *
 * Keeps a libp2p Host going (administering the work thread with m_workNet).
 *
 * Encapsulates a bunch of P2P protocols (interfaces), each using the same underlying libp2p Host.
 *
 * Provides a baseline for the multiplexed multi-protocol session class, WebThree.
 */
class WebThreeDirect : public WebThreeNetworkFace
{
public:
	/// Constructor for private instance. If there is already another process on the machine using @a _dbPath, then this will throw an exception.
	/// ethereum() may be safely static_cast()ed to a eth::Client*.
	WebThreeDirect(
		std::string const& _clientVersion,
		std::string const& _dbPath,
		WithExisting _we = WithExisting::Trust,
		std::set<std::string> const& _interfaces = {"eth", "shh"},
		p2p::NetworkPreferences const& _n = p2p::NetworkPreferences(),
		bytesConstRef _network = bytesConstRef()
	);

	/// Destructor.
	~WebThreeDirect();

	// The mainline interfaces:

	eth::Client* ethereum() const { if (!m_ethereum) BOOST_THROW_EXCEPTION(InterfaceNotSupported("eth")); return m_ethereum.get(); }
	std::shared_ptr<shh::WhisperHost> whisper() const { auto w = m_whisper.lock(); if (!w) BOOST_THROW_EXCEPTION(InterfaceNotSupported("shh")); return w; }
	bzz::Interface* swarm() const { BOOST_THROW_EXCEPTION(InterfaceNotSupported("bzz")); }

	// Misc stuff:

	static std::string composeClientVersion(std::string const& _client, std::string const& _name);

	std::string const& clientVersion() const { return m_clientVersion; }

	void setClientVersion(std::string const& _name) { m_clientVersion = _name; }

	// Network stuff:

	/// Get information on the current peer set.
	std::vector<p2p::PeerSessionInfo> peers() override;

	/// Same as peers().size(), but more efficient.
	size_t peerCount() const override;
	
	/// Add node to connect to.
	virtual void addNode(p2p::NodeId const& _node, bi::tcp::endpoint const& _hostEndpoint) override;
	
	/// Add node to connect to.
	void addNode(p2p::NodeId const& _node, std::string const& _hostString) { addNode(_node, p2p::Network::resolveHost(_hostString)); }
	
	/// Add node to connect to.
	void addNode(bi::tcp::endpoint const& _endpoint) { addNode(p2p::NodeId(), _endpoint); }

	/// Add node to connect to.
	void addNode(std::string const& _hostString) { addNode(p2p::NodeId(), _hostString); }
	
	/// Require connection to peer.
	void requirePeer(p2p::NodeId const& _node, bi::tcp::endpoint const& _endpoint) override;

	/// Require connection to peer.
	void requirePeer(p2p::NodeId const& _node, std::string const& _hostString) { requirePeer(_node, p2p::Network::resolveHost(_hostString)); }

	/// Save peers
	dev::bytes saveNetwork() override;

	/// Sets the ideal number of peers.
	void setIdealPeerCount(size_t _n) override;

	bool haveNetwork() const override { return m_net.haveNetwork(); }

	p2p::NetworkPreferences const& networkPreferences() const override;

	void setNetworkPreferences(p2p::NetworkPreferences const& _n, bool _dropPeers = false) override;

	p2p::NodeInfo nodeInfo() const override { return m_net.nodeInfo(); }

	p2p::NodeId id() const override { return m_net.id(); }

	std::string enode() const override { return m_net.enode(); }

	/// Gets the nodes.
	p2p::Peers nodes() const override { return m_net.getPeers(); }

	/// Start the network subsystem.
	void startNetwork() override { m_net.start(); }

	/// Stop the network subsystem.
	void stopNetwork() override { m_net.stop(); }

	/// Is network working? there may not be any peers yet.
	bool isNetworkStarted() const override { return m_net.isStarted(); }

private:
	std::string m_clientVersion;					///< Our end-application client's name/version.

	p2p::Host m_net;								///< Should run in background and send us events when blocks found and allow us to send blocks as required.

	std::unique_ptr<eth::Client> m_ethereum;		///< Main interface for Ethereum ("eth") protocol.
	std::weak_ptr<shh::WhisperHost> m_whisper;		///< Main interface for Whisper ("shh") protocol.
};



// TODO, probably move into libdevrpc:

class RPCSlave {};
class RPCMaster {};

// TODO, probably move into eth:

class EthereumSlave: public eth::Interface
{
public:
	EthereumSlave(RPCSlave*) {}

	// TODO: implement all of the virtuals with the RLPClient link.
};

class EthereumMaster
{
public:
	EthereumMaster(RPCMaster*) {}

	// TODO: implement the master-end of whatever the RLPClient link will send over.
};

// TODO, probably move into shh:

class WhisperSlave: public shh::Interface
{
public:
	WhisperSlave(RPCSlave*) {}

	// TODO: implement all of the virtuals with the RLPClient link.
};

class WhisperMaster
{
public:
	WhisperMaster(RPCMaster*) {}

	// TODO: implement the master-end of whatever the RLPClient link will send over.
};

/**
 * @brief Main API hub for interfacing with Web 3 components.
 *
 * This does transparent local multiplexing, so you can have as many running on the
 * same machine all working from a single DB path.
 */
class WebThree
{
public:
	/// Constructor for public instance. This will be shared across the local machine.
	WebThree();

	/// Destructor.
	~WebThree();

	// The mainline interfaces.

	eth::Interface* ethereum() const { if (!m_ethereum) BOOST_THROW_EXCEPTION(InterfaceNotSupported("eth")); return m_ethereum; }
	shh::Interface* whisper() const { if (!m_whisper) BOOST_THROW_EXCEPTION(InterfaceNotSupported("shh")); return m_whisper; }
	bzz::Interface* swarm() const { BOOST_THROW_EXCEPTION(InterfaceNotSupported("bzz")); }

	// Peer network stuff - forward through RPCSlave, probably with P2PNetworkSlave/Master classes like Whisper & Ethereum.

	/// Get information on the current peer set.
	std::vector<p2p::PeerSessionInfo> peers();

	/// Same as peers().size(), but more efficient.
	size_t peerCount() const;

	/// Connect to a particular peer.
	void connect(std::string const& _seedHost, unsigned short _port = 30303);

	/// Is the network subsystem up?
	bool haveNetwork();

	/// Save peers
	dev::bytes savePeers();

	/// Restore peers
	void restorePeers(bytesConstRef _saved);

private:
	EthereumSlave* m_ethereum = nullptr;
	WhisperSlave* m_whisper = nullptr;

	// TODO:
	RPCSlave m_rpcSlave;
};

}
