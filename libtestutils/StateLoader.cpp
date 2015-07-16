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
/** @file StateLoader.cpp
 * @author Marek Kotewicz <marek@ethdev.com>
 * @date 2015
 */

#include "StateLoader.h"

using namespace std;
using namespace dev;
using namespace dev::eth;
using namespace dev::test;

StateLoader::StateLoader(Json::Value const& _json, std::string const& _dbPath):
	m_state(State::openDB(_dbPath, h256{}, WithExisting::Kill), BaseState::Empty)
{
	for (string const& name: _json.getMemberNames())
	{
		Json::Value o = _json[name];

		Address address = Address(name);
		bytes code = fromHex(o["code"].asString().substr(2));

		if (!code.empty())
		{
			m_state.m_cache[address] = Account(u256(o["balance"].asString()), Account::ContractConception);
			m_state.m_cache[address].setCode(std::move(code));
		}
		else
			m_state.m_cache[address] = Account(u256(o["balance"].asString()), Account::NormalCreation);

		for (string const& j: o["storage"].getMemberNames())
			m_state.setStorage(address, u256(j), u256(o["storage"][j].asString()));

		for (auto i = 0; i < u256(o["nonce"].asString()); ++i)
			m_state.noteSending(address);

		m_state.ensureCached(address, false, false);
	}

	m_state.commit();
}
