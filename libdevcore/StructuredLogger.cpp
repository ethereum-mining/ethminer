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
/** @file StructuredLogger.h
 * @author Lefteris Karapetsas <lefteris@ethdev.com>
 * @date 2015
 *
 * A simple helper class for the structured logging
 */

#include "StructuredLogger.h"
#include <boost/asio/ip/tcp.hpp>
#if ETH_JSONRPC
#include <json/json.h>
#endif
#include <libdevcore/CommonIO.h>
#include "Guards.h"

namespace ba = boost::asio;
using namespace std;

namespace dev
{

void StructuredLogger::initialize(bool _enabled, std::string const& _timeFormat, std::string const& _destinationURL)
{
	m_enabled = _enabled;
	m_timeFormat = _timeFormat;
	if (_destinationURL.size() > 7 && _destinationURL.substr(0, 7) == "file://")
		m_out.open(_destinationURL.substr(7));
	// TODO: support tcp://
}

void StructuredLogger::outputJson(Json::Value const& _value, std::string const& _name) const
{
#if ETH_JSONRPC
	Json::Value event;
	static Mutex s_lock;
	Json::FastWriter fastWriter;
	Guard l(s_lock);
	event[_name] = _value;
	(m_out.is_open() ? m_out : cout) << fastWriter.write(event) << endl;
#else
	(void)_value;
	(void)_name;
#endif
}

void StructuredLogger::starting(string const& _clientImpl, const char* _ethVersion)
{
#if ETH_JSONRPC
	if (get().m_enabled)
	{
		Json::Value event;
		event["client_impl"] = _clientImpl;
		event["eth_version"] = std::string(_ethVersion);
		// TODO net_version
		event["ts"] = dev::toString(chrono::system_clock::now(), get().m_timeFormat.c_str());

		get().outputJson(event, "starting");
	}
#else
	(void)_clientImpl;
	(void)_ethVersion;
#endif
}

void StructuredLogger::stopping(string const& _clientImpl, const char* _ethVersion)
{
#if ETH_JSONRPC
	if (get().m_enabled)
	{
		Json::Value event;
		event["client_impl"] = _clientImpl;
		event["eth_version"] = std::string(_ethVersion);
		// TODO net_version
		event["ts"] = dev::toString(chrono::system_clock::now(), get().m_timeFormat.c_str());

		get().outputJson(event, "stopping");
	}
#else
	(void)_clientImpl;
	(void)_ethVersion;
#endif
}

void StructuredLogger::p2pConnected(
	string const& _id,
	bi::tcp::endpoint const& _addr,
	chrono::system_clock::time_point const& _ts,
	string const& _remoteVersion,
	unsigned int _numConnections)
{
#if ETH_JSONRPC
	if (get().m_enabled)
	{
		std::stringstream addrStream;
		addrStream << _addr;
		Json::Value event;
		event["remote_version_string"] = _remoteVersion;
		event["remote_addr"] = addrStream.str();
		event["remote_id"] = _id;
		event["num_connections"] = Json::Value(_numConnections);
		event["ts"] = dev::toString(_ts, get().m_timeFormat.c_str());

		get().outputJson(event, "p2p.connected");
	}
#else
	(void)_id;
	(void)_addr;
	(void)_ts;
	(void)_remoteVersion;
	(void)_numConnections;
#endif
}

void StructuredLogger::p2pDisconnected(string const& _id, bi::tcp::endpoint const& _addr, unsigned int _numConnections)
{
#if ETH_JSONRPC
	if (get().m_enabled)
	{
		std::stringstream addrStream;
		addrStream << _addr;
		Json::Value event;
		event["remote_addr"] = addrStream.str();
		event["remote_id"] = _id;
		event["num_connections"] = Json::Value(_numConnections);
		event["ts"] = dev::toString(chrono::system_clock::now(), get().m_timeFormat.c_str());

		get().outputJson(event, "p2p.disconnected");
	}
#else
	(void)_id;
	(void)_addr;
	(void)_numConnections;
#endif
}

void StructuredLogger::minedNewBlock(
	string const& _hash,
	string const& _blockNumber,
	string const& _chainHeadHash,
	string const& _prevHash)
{
#if ETH_JSONRPC
	if (get().m_enabled)
	{
		Json::Value event;
		event["block_hash"] = _hash;
		event["block_number"] = _blockNumber;
		event["chain_head_hash"] = _chainHeadHash;
		event["ts"] = dev::toString(chrono::system_clock::now(), get().m_timeFormat.c_str());
		event["block_prev_hash"] = _prevHash;

		get().outputJson(event, "eth.miner.new_block");
	}
#else
	(void)_hash;
	(void)_blockNumber;
	(void)_chainHeadHash;
	(void)_prevHash;
#endif
}

void StructuredLogger::chainReceivedNewBlock(
	string const& _hash,
	string const& _blockNumber,
	string const& _chainHeadHash,
	string const& _remoteID,
	string const& _prevHash)
{
#if ETH_JSONRPC
	if (get().m_enabled)
	{
		Json::Value event;
		event["block_hash"] = _hash;
		event["block_number"] = _blockNumber;
		event["chain_head_hash"] = _chainHeadHash;
		event["remote_id"] = _remoteID;
		event["ts"] = dev::toString(chrono::system_clock::now(), get().m_timeFormat.c_str());
		event["block_prev_hash"] = _prevHash;

		get().outputJson(event, "eth.chain.received.new_block");
	}
#else
	(void)_hash;
	(void)_blockNumber;
	(void)_chainHeadHash;
	(void)_remoteID;
	(void)_prevHash;
#endif
}

void StructuredLogger::chainNewHead(
	string const& _hash,
	string const& _blockNumber,
	string const& _chainHeadHash,
	string const& _prevHash)
{
#if ETH_JSONRPC
	if (get().m_enabled)
	{
		Json::Value event;
		event["block_hash"] = _hash;
		event["block_number"] = _blockNumber;
		event["chain_head_hash"] = _chainHeadHash;
		event["ts"] = dev::toString(chrono::system_clock::now(), get().m_timeFormat.c_str());
		event["block_prev_hash"] = _prevHash;

		get().outputJson(event, "eth.miner.new_block");
	}
#else
	(void)_hash;
	(void)_blockNumber;
	(void)_chainHeadHash;
	(void)_prevHash;
#endif
}

void StructuredLogger::transactionReceived(string const& _hash, string const& _remoteId)
{
#if ETH_JSONRPC
	if (get().m_enabled)
	{
		Json::Value event;
		event["tx_hash"] = _hash;
		event["remote_id"] = _remoteId;
		event["ts"] = dev::toString(chrono::system_clock::now(), get().m_timeFormat.c_str());

		get().outputJson(event, "eth.tx.received");
	}
#else
	(void)_hash;
	(void)_remoteId;
#endif
}

}
