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
/** @file StateLoader.h
 * @author Marek Kotewicz <marek@ethdev.com>
 * @date 2015
 */

#pragma once

#include <json/json.h>
#include <libdevcore/TransientDirectory.h>
#include <libethereum/State.h>
#include <libethereum/BlockChain.h>

namespace dev
{
namespace test
{

/**
 * @brief Friend of State, loads State from given JSON object
 */
class StateLoader
{
public:
	StateLoader(Json::Value const& _json, std::string const& _dbPath);
	eth::State const& state() const { return m_state; }
	eth::StateDefinition const& stateDefinition() const { return m_state.m_cache; }

private:
	eth::State m_state;
};

}
}
