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
#include <vector>

#include "TestHelper.h"

namespace dev
{
namespace test
{

class Stats: public Listener
{
public:
	using clock = std::chrono::high_resolution_clock;

	struct Item
	{
		clock::duration duration;
		std::string 	name;
	};

	static Stats& get();

	~Stats();

	void suiteStarted(std::string const& _name) override;
	void testStarted(std::string const& _name) override;
	void testFinished() override;

private:
	clock::time_point m_tp;
	std::string m_currentSuite;
	std::string m_currentTest;
	std::vector<Item> m_stats;
};

}
}
