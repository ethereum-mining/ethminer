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
/** @file Web3Server.h.cpp
 * @author Arkadiy Paronyan arkadiy@ethdev.com
 * @date 2014
 * Ethereum IDE client.
 */

#include <libdevcore/Exceptions.h>
#include <libdevcore/Log.h>
#include <libethereum/Interface.h>
#include <libwebthree/WebThree.h>
#include <libweb3jsonrpc/AccountHolder.h>
#include "Web3Server.h"

using namespace dev::mix;
using namespace dev;

namespace
{
class EmptyNetwork : public dev::WebThreeNetworkFace
{
	std::vector<p2p::PeerSessionInfo> peers() override
	{
		return std::vector<p2p::PeerSessionInfo>();
	}

	size_t peerCount() const override
	{
		return 0;
	}

	void addNode(p2p::NodeId const& _node, bi::tcp::endpoint const& _hostEndpoint) override
	{
		(void)_node;
		(void)_hostEndpoint;
	}

	void requirePeer(p2p::NodeId const& _node, bi::tcp::endpoint const& _endpoint) override
	{
		(void)_node;
		(void)_endpoint;
	}

	dev::bytes saveNetwork() override
	{
		return dev::bytes();
	}

	void setIdealPeerCount(size_t _n) override
	{
		(void)_n;
	}

	bool haveNetwork() const override
	{
		return false;
	}

	p2p::NetworkPreferences const& networkPreferences() const override
	{
		static const p2p::NetworkPreferences c_ret;
		return c_ret;
	}

	void setNetworkPreferences(p2p::NetworkPreferences const& _n, bool _dropPeers) override
	{
		(void)_n;
		(void)_dropPeers;
	}

	p2p::NodeInfo nodeInfo() const override { return p2p::NodeInfo(); }

	std::string enode() const override { return ""; }

	p2p::NodeId id() const override
	{
		return p2p::NodeId();
	}

	p2p::Peers nodes() const override
	{
		return p2p::Peers();
	}

	void startNetwork() override
	{
	}

	void stopNetwork() override
	{
	}

	bool isNetworkStarted() const override
	{
		return false;
	}
};

}

Web3Server::Web3Server(jsonrpc::AbstractServerConnector& _conn, std::shared_ptr<eth::AccountHolder> const& _ethAccounts, std::vector<dev::KeyPair> const& _shhAccounts, dev::eth::Interface* _client):
	WebThreeStubServerBase(_conn, _ethAccounts, _shhAccounts),
	m_client(_client),
	m_network(new EmptyNetwork())
{
}

Web3Server::~Web3Server()
{
}

std::shared_ptr<dev::shh::Interface> Web3Server::face()
{
	BOOST_THROW_EXCEPTION(InterfaceNotSupported("dev::shh::Interface"));
}

dev::WebThreeNetworkFace* Web3Server::network()
{
	return m_network.get();
}

std::string Web3Server::get(std::string const& _name, std::string const& _key)
{
	std::string k(_name + "/" + _key);
	return m_db[k];
}

void Web3Server::put(std::string const& _name, std::string const& _key, std::string const& _value)
{
	std::string k(_name + "/" + _key);
	m_db[k] = _value;
}

Json::Value Web3Server::eth_getFilterChanges(std::string const& _filterId)
{
	return WebThreeStubServerBase::eth_getFilterChanges(_filterId);
}

std::string Web3Server::eth_sendTransaction(Json::Value const& _json)
{
	std::string ret = WebThreeStubServerBase::eth_sendTransaction(_json);
	emit newTransaction();
	return ret;
}

std::string Web3Server::eth_call(Json::Value const& _json, std::string const& _blockNumber)
{
	std::string ret = WebThreeStubServerBase::eth_call(_json, _blockNumber);
	emit newTransaction();
	return ret;
}
