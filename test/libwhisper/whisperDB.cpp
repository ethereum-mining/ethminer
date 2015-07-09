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
/** @file whisperMessage.cpp
* @author Vladislav Gluhovsky <vlad@ethdev.com>
* @date July 2015
*/

#include <boost/test/unit_test.hpp>
#include <libwhisper/WhisperDB.h>

using namespace std;
using namespace dev;
using namespace dev::shh;

BOOST_AUTO_TEST_SUITE(whisperDB)

BOOST_AUTO_TEST_CASE(first)
{
	VerbosityHolder setTemporaryLevel(10);
	cnote << "Testing Whisper DB...";

	WhisperDB db;

	h256 h1(0x12345678);
	h256 h2(0xBADD00DE);

	string s;
	string const text1 = "lorem_ipsum";
	string const text2 = "dolor_sit_amet";

	db.kill(h1);
	db.kill(h2);

	db.insert(h1, text2);
	s = db.lookup(h2);
	BOOST_REQUIRE(s.empty());
	s = db.lookup(h1);
	BOOST_REQUIRE(!s.compare(text2));

	db.insert(h1, text1);
	s = db.lookup(h2);
	BOOST_REQUIRE(s.empty());
	s = db.lookup(h1);
	BOOST_REQUIRE(!s.compare(text1));

	db.insert(h2, text2);
	s = db.lookup(h2);
	BOOST_REQUIRE(!s.compare(text2));
	s = db.lookup(h1);
	BOOST_REQUIRE(!s.compare(text1));

	db.kill(h1);
	db.kill(h2);
}

BOOST_AUTO_TEST_SUITE_END()
