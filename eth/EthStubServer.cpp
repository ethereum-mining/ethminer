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
/** @file EthStubServer.cpp
 * @authors:
 *   Gav Wood <i@gavwood.com>
 * @date 2014
 */

#if ETH_JSONRPC
#include "EthStubServer.h"
#include <libevmface/Instruction.h>
#include <liblll/Compiler.h>
#include <libethereum/Client.h>
#include <libwebthree/WebThree.h>
#include <libdevcore/CommonJS.h>
using namespace std;
using namespace dev;
using namespace dev::eth;

EthStubServer::EthStubServer(jsonrpc::AbstractServerConnector* _conn, WebThreeDirect& _web3):
	AbstractEthStubServer(_conn),
	m_web3(_web3)
{
}

dev::eth::Client& EthStubServer::ethereum() const
{
	return *m_web3.ethereum();
}

std::string EthStubServer::balanceAt(const string &a, const string &block)
{

}

std::string EthStubServer::block(const string &numberOrHash)
{

}

std::string EthStubServer::call(const string &json)
{

}

std::string EthStubServer::codeAt(const string &a, const string &block)
{

}

std::string EthStubServer::coinbase()
{

}

int EthStubServer::countAt(const string &a, const string &block)
{

}

int EthStubServer::defaultBlock()
{

}

std::string EthStubServer::fromAscii(const string &s)
{

}

std::string EthStubServer::fromFixed(const string &s)
{

}

std::string EthStubServer::gasPrice()
{

}

bool EthStubServer::isListening()
{

}

bool EthStubServer::isMining()
{

}

std::string EthStubServer::key()
{
    if (!m_keys.size())
        return std::string();
    return toJS(m_keys[0].sec());
}

Json::Value EthStubServer::keys()
{
    Json::Value ret;
    for (auto i: m_keys)
        ret.append(toJS(i.secret()));
    return ret;
}

std::string EthStubServer::lll(const string &s)
{

}

std::string EthStubServer::messages(const string &json)
{

}

int EthStubServer::number()
{

}

int EthStubServer::peerCount()
{
    return m_web3.peerCount();
}

std::string EthStubServer::secretToAddress(const string &s)
{

}

std::string EthStubServer::setListening(const string &l)
{

}

std::string EthStubServer::setMining(const string &l)
{

}

std::string EthStubServer::sha3(const string &s)
{

}

std::string EthStubServer::stateAt(const string &a, const string &block, const string &p)
{

}

std::string EthStubServer::toAscii(const string &s)
{

}

std::string EthStubServer::toDecimal(const string &s)
{

}

std::string EthStubServer::toFixed(const string &s)
{

}

std::string EthStubServer::transact(const string &json)
{

}

std::string EthStubServer::transaction(const string &i, const string &numberOrHash)
{

}

std::string EthStubServer::uncle(const string &i, const string &numberOrHash)
{

}

std::string EthStubServer::watch(const string &json)
{

}

Json::Value EthStubServer::blockJson(const std::string& _hash)
{
	Json::Value res;
	auto const& bc = ethereum().blockChain();
	
	auto b = _hash.length() ? bc.block(h256(_hash)) : bc.block();
	
	auto bi = BlockInfo(b);
	res["number"] = boost::lexical_cast<string>(bi.number);
	res["hash"] = boost::lexical_cast<string>(bi.hash);
	res["parentHash"] = boost::lexical_cast<string>(bi.parentHash);
	res["sha3Uncles"] = boost::lexical_cast<string>(bi.sha3Uncles);
	res["coinbaseAddress"] = boost::lexical_cast<string>(bi.coinbaseAddress);
	res["stateRoot"] = boost::lexical_cast<string>(bi.stateRoot);
	res["transactionsRoot"] = boost::lexical_cast<string>(bi.transactionsRoot);
	res["minGasPrice"] = boost::lexical_cast<string>(bi.minGasPrice);
	res["gasLimit"] = boost::lexical_cast<string>(bi.gasLimit);
	res["gasUsed"] = boost::lexical_cast<string>(bi.gasUsed);
	res["difficulty"] = boost::lexical_cast<string>(bi.difficulty);
	res["timestamp"] = boost::lexical_cast<string>(bi.timestamp);
	res["nonce"] = boost::lexical_cast<string>(bi.nonce);

	return res;
}

Json::Value EthStubServer::jsontypeToValue(int _jsontype)
{
	switch (_jsontype)
	{
		case jsonrpc::JSON_STRING: return ""; //Json::stringValue segfault, fuck knows why
		case jsonrpc::JSON_BOOLEAN: return Json::booleanValue;
		case jsonrpc::JSON_INTEGER: return Json::intValue;
		case jsonrpc::JSON_REAL: return Json::realValue;
		case jsonrpc::JSON_OBJECT: return Json::objectValue;
		case jsonrpc::JSON_ARRAY: return Json::arrayValue;
		default: return Json::nullValue;
	}
}

#endif
