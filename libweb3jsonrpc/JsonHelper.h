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
/** @file JsonHelper.h
 * @authors:
 *   Gav Wood <i@gavwood.com>
 * @date 2015
 */
#pragma once

#include <json/json.h>
#include <libethcore/Common.h>
#include <libethcore/BlockInfo.h>
#include <libethereum/LogFilter.h>
#include <libwhisper/Message.h>

namespace dev
{

Json::Value toJson(std::map<u256, u256> const& _storage);
Json::Value toJson(std::unordered_map<u256, u256> const& _storage);

namespace p2p
{

Json::Value toJson(PeerSessionInfo const& _p);

}

namespace eth
{

class Transaction;
class LocalisedTransaction;
struct BlockDetails;
class Interface;
using Transactions = std::vector<Transaction>;
using UncleHashes = h256s;
using TransactionHashes = h256s;

Json::Value toJson(BlockInfo const& _bi);
//TODO: wrap these params into one structure eg. "LocalisedTransaction"
Json::Value toJson(Transaction const& _t, std::pair<h256, unsigned> _location, BlockNumber _blockNumber);
Json::Value toJson(BlockInfo const& _bi, BlockDetails const& _bd, UncleHashes const& _us, Transactions const& _ts);
Json::Value toJson(BlockInfo const& _bi, BlockDetails const& _bd, UncleHashes const& _us, TransactionHashes const& _ts);
Json::Value toJson(TransactionSkeleton const& _t);
Json::Value toJson(Transaction const& _t);
Json::Value toJson(LocalisedTransaction const& _t);
Json::Value toJson(TransactionReceipt const& _t);
Json::Value toJson(LocalisedTransactionReceipt const& _t);
Json::Value toJson(LocalisedLogEntry const& _e);
Json::Value toJson(LogEntry const& _e);
Json::Value toJson(std::unordered_map<h256, LocalisedLogEntries> const& _entriesByBlock);
Json::Value toJsonByBlock(LocalisedLogEntries const& _entries);
TransactionSkeleton toTransactionSkeleton(Json::Value const& _json);
LogFilter toLogFilter(Json::Value const& _json);
LogFilter toLogFilter(Json::Value const& _json, Interface const& _client);	// commented to avoid warning. Uncomment once in use @ PoC-7.

template <class BlockInfoSub>
Json::Value toJson(BlockHeaderPolished<BlockInfoSub> const& _bh)
{
	Json::Value res;
	if (_bh)
	{
		res = toJson(static_cast<BlockInfo const&>(_bh));
		for (auto const& i: _bh.jsInfo())
			res[i.first] = i.second;
	}
	return res;
}

}

namespace shh
{

Json::Value toJson(h256 const& _h, Envelope const& _e, Message const& _m);
Message toMessage(Json::Value const& _json);
Envelope toSealed(Json::Value const& _json, Message const& _m, Secret const& _from);
std::pair<Topics, Public> toWatch(Json::Value const& _json);

}

template <class T>
Json::Value toJson(std::vector<T> const& _es)
{
	Json::Value res(Json::arrayValue);
	for (auto const& e: _es)
		res.append(toJson(e));
	return res;
}

template <class T>
Json::Value toJson(std::unordered_set<T> const& _es)
{
	Json::Value res(Json::arrayValue);
	for (auto const& e: _es)
		res.append(toJson(e));
	return res;
}

template <class T>
Json::Value toJson(std::set<T> const& _es)
{
	Json::Value res(Json::arrayValue);
	for (auto const& e: _es)
		res.append(toJson(e));
	return res;
}

}
