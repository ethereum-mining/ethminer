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

#pragma once

#include <chrono>
#include <map>

#include "TestHelper.h"

namespace dev
{
namespace test
{

class Stats: public Listener
{
public:
	using clock = std::chrono::high_resolution_clock;
	using stats_t = std::map<clock::duration, std::string>;

	static Stats& get();

	~Stats();

	void testStarted(std::string const& _name) override;
	void testFinished() override;

private:
	clock::time_point m_tp;
	std::string m_currentTest;
	stats_t m_stats;
};

}
}
