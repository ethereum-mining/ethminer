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
* @date June 2015
*/

#include <boost/test/unit_test.hpp>
#include <libwhisper/BloomFilter.h>

using namespace std;
using namespace dev;
using namespace dev::shh;

BOOST_AUTO_TEST_SUITE(bloomFilter)

BOOST_AUTO_TEST_CASE(match)
{
	VerbosityHolder setTemporaryLevel(10);
	cnote << "Testing Bloom Filter matching...";

	SharedBloomFilter f;
	unsigned b00000001 = 0x01;
	unsigned b00010000 = 0x10;
	unsigned b00011000 = 0x18;
	unsigned b00110000 = 0x30;
	unsigned b00110010 = 0x32;
	unsigned b00111000 = 0x38;
	unsigned b00000110 = 0x06;
	unsigned b00110110 = 0x36;
	unsigned b00110111 = 0x37;

	BOOST_REQUIRE(!f.matches(AbridgedTopic(b00000001)));
	f.add(AbridgedTopic(b00000001));
	BOOST_REQUIRE(!f.matches(AbridgedTopic(b00010000)));
	f.add(AbridgedTopic(b00010000));
	BOOST_REQUIRE(!f.matches(AbridgedTopic(b00011000)));
	f.add(AbridgedTopic(b00011000));
	BOOST_REQUIRE(!f.matches(AbridgedTopic(b00110000)));
	f.add(AbridgedTopic(b00110000));
	BOOST_REQUIRE(f.matches(AbridgedTopic(b00111000)));
	BOOST_REQUIRE(!f.matches(AbridgedTopic(b00110010)));
	f.add(AbridgedTopic(b00110010));
	BOOST_REQUIRE(!f.matches(AbridgedTopic(b00000110)));
	f.add(AbridgedTopic(b00000110));

	BOOST_REQUIRE(f.matches(AbridgedTopic(b00110110)));
	BOOST_REQUIRE(f.matches(AbridgedTopic(b00110111)));

	f.remove(AbridgedTopic(b00000001));
	f.remove(AbridgedTopic(b00000001));
	f.remove(AbridgedTopic(b00000001));
	BOOST_REQUIRE(!f.matches(AbridgedTopic(b00000001)));
	BOOST_REQUIRE(f.matches(AbridgedTopic(b00010000)));
	BOOST_REQUIRE(f.matches(AbridgedTopic(b00011000)));
	BOOST_REQUIRE(f.matches(AbridgedTopic(b00110000)));
	BOOST_REQUIRE(f.matches(AbridgedTopic(b00110010)));
	BOOST_REQUIRE(f.matches(AbridgedTopic(b00111000)));
	BOOST_REQUIRE(f.matches(AbridgedTopic(b00000110)));
	BOOST_REQUIRE(f.matches(AbridgedTopic(b00110110)));
	BOOST_REQUIRE(!f.matches(AbridgedTopic(b00110111)));

	f.remove(AbridgedTopic(b00010000));
	BOOST_REQUIRE(!f.matches(AbridgedTopic(b00000001)));
	BOOST_REQUIRE(f.matches(AbridgedTopic(b00010000)));
	BOOST_REQUIRE(f.matches(AbridgedTopic(b00011000)));
	BOOST_REQUIRE(f.matches(AbridgedTopic(b00110000)));
	BOOST_REQUIRE(f.matches(AbridgedTopic(b00110010)));
	BOOST_REQUIRE(f.matches(AbridgedTopic(b00111000)));
	BOOST_REQUIRE(f.matches(AbridgedTopic(b00000110)));
	BOOST_REQUIRE(f.matches(AbridgedTopic(b00110110)));
	BOOST_REQUIRE(!f.matches(AbridgedTopic(b00110111)));

	f.remove(AbridgedTopic(b00111000));
	BOOST_REQUIRE(!f.matches(AbridgedTopic(b00000001)));
	BOOST_REQUIRE(f.matches(AbridgedTopic(b00010000)));
	BOOST_REQUIRE(!f.matches(AbridgedTopic(b00011000)));
	BOOST_REQUIRE(f.matches(AbridgedTopic(b00110000)));
	BOOST_REQUIRE(f.matches(AbridgedTopic(b00110010)));
	BOOST_REQUIRE(!f.matches(AbridgedTopic(b00111000)));
	BOOST_REQUIRE(f.matches(AbridgedTopic(b00000110)));
	BOOST_REQUIRE(f.matches(AbridgedTopic(b00110110)));
	BOOST_REQUIRE(!f.matches(AbridgedTopic(b00110111)));

	f.add(AbridgedTopic(b00000001));
	BOOST_REQUIRE(f.matches(AbridgedTopic(b00000001)));
	BOOST_REQUIRE(f.matches(AbridgedTopic(b00010000)));
	BOOST_REQUIRE(!f.matches(AbridgedTopic(b00011000)));
	BOOST_REQUIRE(f.matches(AbridgedTopic(b00110000)));
	BOOST_REQUIRE(f.matches(AbridgedTopic(b00110010)));
	BOOST_REQUIRE(!f.matches(AbridgedTopic(b00111000)));
	BOOST_REQUIRE(f.matches(AbridgedTopic(b00000110)));
	BOOST_REQUIRE(f.matches(AbridgedTopic(b00110110)));
	BOOST_REQUIRE(f.matches(AbridgedTopic(b00110111)));

	f.remove(AbridgedTopic(b00110111));
	BOOST_REQUIRE(!f.matches(AbridgedTopic(b00000001)));
	BOOST_REQUIRE(f.matches(AbridgedTopic(b00010000)));
	BOOST_REQUIRE(!f.matches(AbridgedTopic(b00011000)));
	BOOST_REQUIRE(!f.matches(AbridgedTopic(b00110000)));
	BOOST_REQUIRE(!f.matches(AbridgedTopic(b00110010)));
	BOOST_REQUIRE(!f.matches(AbridgedTopic(b00111000)));
	BOOST_REQUIRE(!f.matches(AbridgedTopic(b00000110)));
	BOOST_REQUIRE(!f.matches(AbridgedTopic(b00110110)));
	BOOST_REQUIRE(!f.matches(AbridgedTopic(b00110111)));

	f.remove(AbridgedTopic(b00110111));
	BOOST_REQUIRE(!f.matches(AbridgedTopic(b00000001)));
	BOOST_REQUIRE(!f.matches(AbridgedTopic(b00010000)));
	BOOST_REQUIRE(!f.matches(AbridgedTopic(b00011000)));
	BOOST_REQUIRE(!f.matches(AbridgedTopic(b00110000)));
	BOOST_REQUIRE(!f.matches(AbridgedTopic(b00110010)));
	BOOST_REQUIRE(!f.matches(AbridgedTopic(b00111000)));
	BOOST_REQUIRE(!f.matches(AbridgedTopic(b00000110)));
	BOOST_REQUIRE(!f.matches(AbridgedTopic(b00110110)));
	BOOST_REQUIRE(!f.matches(AbridgedTopic(b00110111)));
}

BOOST_AUTO_TEST_SUITE_END()