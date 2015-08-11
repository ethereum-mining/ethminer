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
/** @file difficulty.cpp
 * @author Christoph Jentzsch <cj@ethdev.com>
 * @date 2015
 * difficulty calculation tests.
 */


#include <boost/test/unit_test.hpp>
#include <test/TestHelper.h>
#include <libethcore/BlockInfo.h>

using namespace std;
using namespace dev;
using namespace dev::eth;

namespace js = json_spirit;

BOOST_AUTO_TEST_SUITE(DifficultyTests)

BOOST_AUTO_TEST_CASE(difficultyTests)
{
	string testPath = test::getTestPath();
	testPath += "/BasicTests";

	js::mValue v;
	string s = contentsString(testPath + "/difficulty.json");
	BOOST_REQUIRE_MESSAGE(s.length() > 0, "Contents of 'difficulty.json' is empty. Have you cloned the 'tests' repo branch develop?");
	js::read_string(s, v);

	for (auto& i: v.get_obj())
	{
		js::mObject o = i.second.get_obj();
		cnote << "Difficulty test: " << i.first;
		BlockInfo parent;
		parent.setTimestamp(test::toInt(o["parentTimestamp"]));
		parent.setDifficulty(test::toInt(o["parentDifficulty"]));
		parent.setNumber(test::toInt(o["currentBlockNumber"]) - 1);

		BlockInfo current;
		current.setTimestamp(test::toInt(o["currentTimestamp"]));
		current.setNumber(test::toInt(o["currentBlockNumber"]));

		BOOST_CHECK_EQUAL(current.calculateDifficulty(parent), test::toInt(o["currentDifficulty"]));
	}
}

BOOST_AUTO_TEST_SUITE_END()
