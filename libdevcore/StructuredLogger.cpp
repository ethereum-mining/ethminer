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

#include <json/json.h>

using namespace std;

namespace dev
{

char const* StructuredLogger::timePointToString(chrono::system_clock::time_point const& _ts) const
{
	// not using C++11 std::put_time due to gcc bug
	// http://stackoverflow.com/questions/14136833/stdput-time-implementation-status-in-gcc

	// TODO: Format it according to Log event Requirements
	time_t time = chrono::system_clock::to_time_t(_ts);
	return ctime(&time);
}

void StructuredLogger::outputJson(Json::Value const* _value, std::string const& _name) const
{
	Json::Value event;
	event[_name] = _value;
	cout << event;
}

void StructuredLogger::logStarting(string const& _clientImpl, const char* _ethVersion)
{
	if (m_enabled)
	{
		Json::Value event;
		event["comment"] = "one of the first log events, before any operation is started";
		event["client_implt"] = _clientImpl;
		event["eth_version"] = std::string(_ethVersion);
		event["ts"] = string(timePointToString(std::chrono::system_clock::now()));

		outputJson(&event, "starting");
	}
}

void StructuredLogger::logP2PConnected(string const& _id, bi::tcp::endpoint const& _addr,
	chrono::system_clock::time_point const& _ts, unsigned int _numConnections) const
{
	if (m_enabled)
	{
		std::stringstream addrStream;
		addrStream << _addr;
		Json::Value event;
		event["remote_version_string"] = ""; //TODO
		event["comment"] = "as soon as a successful connection to another node is established";
		event["remote_addr"] = addrStream.str();
		event["remote_id"] = _id;
		event["num_connections"] = Json::Value(_numConnections);
		event["ts"] = string(timePointToString(_ts));

		outputJson(&event, "p2p.connected");
	}
}

void StructuredLogger::logP2PDisconnected(string const& _id, unsigned int _numConnections, bi::tcp::endpoint const& _addr) const
{
	if (m_enabled)
	{
		std::stringstream addrStream;
		addrStream << _addr;
		Json::Value event;
		event["comment"] = "as soon as a disconnection from another node happened";
		event["remote_addr"] = addrStream.str();
		event["remote_id"] = _id;
		event["num_connections"] = Json::Value(_numConnections);
		event["ts"] = string(timePointToString(chrono::system_clock::now()));

		outputJson(&event, "p2p.disconnected");
	}
}

void StructuredLogger::logMinedNewBlock(string const& _hash, string const& _blockNumber,
	string const& _chainHeadHash, string const& _prevHash) const
{
	if (m_enabled)
	{
		Json::Value event;
		event["comment"] = "as soon as the block was mined, before adding as new head";
		event["block_hash"] = _hash;
		event["block_number"] = _blockNumber;
		event["chain_head_hash"] = _chainHeadHash;
		event["ts"] = string(timePointToString(std::chrono::system_clock::now()));
		event["block_prev_hash"] = _prevHash;

		outputJson(&event, "eth.miner.new_block");
	}
}

void StructuredLogger::logChainReceivedNewBlock(string const& _hash, string const& _blockNumber,
	string const& _chainHeadHash, string const& _remoteID, string const& _prevHash) const
{
	if (m_enabled)
	{
		Json::Value event;
		event["comment"] = "whenever a _new_ block is received, before adding";
		event["block_hash"] = _hash;
		event["block_number"] = _blockNumber;
		event["chain_head_hash"] = _chainHeadHash;
		event["remote_id"] = _remoteID;
		event["ts"] = string(timePointToString(chrono::system_clock::now()));
		event["block_prev_hash"] = _prevHash;

		outputJson(&event, "eth.chain.received.new_block");
	}
}

void StructuredLogger::logChainNewHead(string const& _hash, string const& _blockNumber,
	string const& _chainHeadHash, string const& _prevHash) const
{
	if (m_enabled)
	{
		Json::Value event;
		event["comment"] = "whenever head changes";
		event["block_hash"] = _hash;
		event["block_number"] = _blockNumber;
		event["chain_head_hash"] = _chainHeadHash;
		event["ts"] = string(timePointToString(chrono::system_clock::now()));
		event["block_prev_hash"] = _prevHash;

		outputJson(&event, "eth.miner.new_block");
	}
}

void StructuredLogger::logTransactionReceived(string const& _hash, string const& _remoteId) const
{
	if (m_enabled)
	{
		Json::Value event;
		event["tx_hash"] = _hash;
		event["remote_id"] = _remoteId;
		event["ts"] = string(timePointToString(chrono::system_clock::now()));

		outputJson(&event, "eth.tx.received");
	}
}


}
