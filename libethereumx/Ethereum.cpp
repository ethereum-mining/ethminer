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
/** @file Ethereum.cpp
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 */

#include "Ethereum.h"

#include <libethential/Log.h>
#include <libethereum/Client.h>
using namespace std;
using namespace dev;
using namespace dev::eth;
using namespace p2p;

Ethereum::Ethereum()
{
	ensureReady();
}

void Ethereum::ensureReady()
{
	while (!m_client && connectionOpen())
		try
		{
			m_client = unique_ptr<Client>(new Client("+ethereum+"));
			if (m_client)
				startRPCServer();
		}
		catch (DatabaseAlreadyOpen)
		{
			connectToRPCServer();
		}
}

Ethereum::~Ethereum()
{
}

bool Ethereum::connectionOpen() const
{
	return false;
}

void Ethereum::connectToRPCServer()
{
}

void Ethereum::startRPCServer()
{
}

void Ethereum::flushTransactions()
{
}

std::vector<PeerInfo> Ethereum::peers()
{
	return std::vector<PeerInfo>();
}

size_t Ethereum::peerCount() const
{
	return 0;
}

void Ethereum::connect(std::string const& _seedHost, unsigned short _port)
{
}

void Ethereum::submitTransaction(Secret const& _secret, u256 _value, Address _dest, bytes const& _data, u256 _gas, u256 _gasPrice)
{
}

bytes Ethereum::call(Address const& _from, u256 _value, Address _dest, bytes const& _data, u256 _gas, u256 _gasPrice)
{
	return bytes();
}

Address Ethereum::submitTransaction(Secret const& _secret, u256 _endowment, bytes const& _init, u256 _gas, u256 _gasPrice)
{
	return Address();
}

void Ethereum::inject(bytesConstRef _rlp)
{
}

u256 Ethereum::balanceAt(Address _a, int _block) const
{
	return u256();
}

PastMessages Ethereum::messages(MessageFilter const& _filter) const
{
}

std::map<u256, u256> Ethereum::storageAt(Address _a, int _block) const
{
	return std::map<u256, u256>();
}

u256 Ethereum::countAt(Address _a, int _block) const
{
	return u256();
}

u256 Ethereum::stateAt(Address _a, u256 _l, int _block) const
{
	return u256();
}

bytes Ethereum::codeAt(Address _a, int _block) const
{
	return bytes();
}
