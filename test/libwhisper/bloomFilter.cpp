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
#include <libdevcore/SHA3.h>
#include <libwhisper/BloomFilter.h>

using namespace std;
using namespace dev;
using namespace dev::shh;

void testAddNonExisting(TopicBloomFilter& _f, AbridgedTopic const& _h)
{
	BOOST_REQUIRE(!_f.containsRaw(_h));
	_f.addRaw(_h);
	BOOST_REQUIRE(_f.containsRaw(_h));
}

void testRemoveExisting(TopicBloomFilter& _f, AbridgedTopic const& _h)
{
	BOOST_REQUIRE(_f.containsRaw(_h));
	_f.removeRaw(_h);
	BOOST_REQUIRE(!_f.containsRaw(_h));
}

void testAddNonExistingBloom(TopicBloomFilter& _f, AbridgedTopic const& _h)
{
	BOOST_REQUIRE(!_f.containsBloom(_h));
	_f.addBloom(_h);
	BOOST_REQUIRE(_f.containsBloom(_h));
}

void testRemoveExistingBloom(TopicBloomFilter& _f, AbridgedTopic const& _h)
{
	BOOST_REQUIRE(_f.containsBloom(_h));
	_f.removeBloom(_h);
	BOOST_REQUIRE(!_f.containsBloom(_h));
}

int calculateExpected(TopicBloomFilter const& f, int const n)
{
	int const m = f.size * 8; // number of bits in the bloom
	int const k = f.BitsPerBloom; // number of hash functions (e.g. bits set to 1 in every bloom)

	double singleBitSet = 1.0 / m; // probability of any bit being set after inserting a single bit
	double singleBitNotSet = (1.0 - singleBitSet);

	double singleNot = 1; // single bit not set after inserting N elements in the bloom filter
	for (int i = 0; i < k * n; ++i)
		singleNot *= singleBitNotSet;

	double single = 1.0 - singleNot; // probability of a single bit being set after inserting N elements in the bloom filter

	double kBitsSet = 1; // probability of K bits being set after inserting N elements in the bloom filter
	for (int i = 0; i < k; ++i)
		kBitsSet *= single;

	return static_cast<int>(kBitsSet * 100 + 0.5); // in percents, rounded up
}

void testFalsePositiveRate(TopicBloomFilter const& f, int const inserted, Topic& x)
{
	int const c_sampleSize = 1000;
	int falsePositive = 0;

	for (int i = 0; i < c_sampleSize; ++i)
	{
		x = sha3(x);
		AbridgedTopic a(x);
		if (f.containsBloom(a))
			++falsePositive;
	}

	falsePositive /= (c_sampleSize / 100); // in percents
	int expected = calculateExpected(f, inserted);
	int allowed = expected + (expected / 5); // allow deviations ~20%

	//cnote << "Inserted: " << inserted << ", False Positive Rate: " << falsePositive << ", Expected: " << expected;
	BOOST_REQUIRE(falsePositive <= allowed);
}

BOOST_AUTO_TEST_SUITE(bloomFilter)

BOOST_AUTO_TEST_CASE(falsePositiveRate)
{
	VerbosityHolder setTemporaryLevel(10);
	cnote << "Testing Bloom Filter False Positive Rate...";

	TopicBloomFilter f;
	Topic x(0xABCDEF); // deterministic pseudorandom value

	for (int i = 1; i < 21; ++i)
	{
		x = sha3(x);
		f.addBloom(AbridgedTopic(x));
		testFalsePositiveRate(f, i, x);
	}
}

BOOST_AUTO_TEST_CASE(bloomFilterRandom)
{
	VerbosityHolder setTemporaryLevel(10);
	cnote << "Testing Bloom Filter matching...";

	TopicBloomFilter f;
	vector<AbridgedTopic> vec;
	Topic x(0xDEADBEEF);
	int const c_rounds = 4;

	for (int i = 0; i < c_rounds; ++i, x = sha3(x))
		vec.push_back(abridge(x));

	for (int i = 0; i < c_rounds; ++i) 
		testAddNonExisting(f, vec[i]);

	for (int i = 0; i < c_rounds; ++i)
		testRemoveExisting(f, vec[i]);

	for (int i = 0; i < c_rounds; ++i) 
		testAddNonExistingBloom(f, vec[i]);

	for (int i = 0; i < c_rounds; ++i)
		testRemoveExistingBloom(f, vec[i]);
}

BOOST_AUTO_TEST_CASE(bloomFilterRaw)
{
	VerbosityHolder setTemporaryLevel(10);
	cnote << "Testing Raw Bloom matching...";

	TopicBloomFilter f;

	AbridgedTopic b00000001(0x01);
	AbridgedTopic b00010000(0x10);
	AbridgedTopic b00011000(0x18);
	AbridgedTopic b00110000(0x30);
	AbridgedTopic b00110010(0x32);
	AbridgedTopic b00111000(0x38);
	AbridgedTopic b00000110(0x06);
	AbridgedTopic b00110110(0x36);
	AbridgedTopic b00110111(0x37);

	testAddNonExisting(f, b00000001);
	testAddNonExisting(f, b00010000);	
	testAddNonExisting(f, b00011000);	
	testAddNonExisting(f, b00110000);
	BOOST_REQUIRE(f.contains(b00111000));	
	testAddNonExisting(f, b00110010);	
	testAddNonExisting(f, b00000110);
	BOOST_REQUIRE(f.contains(b00110110));
	BOOST_REQUIRE(f.contains(b00110111));

	f.removeRaw(b00000001);
	f.removeRaw(b00000001);
	f.removeRaw(b00000001);
	BOOST_REQUIRE(!f.contains(b00000001));
	BOOST_REQUIRE(f.contains(b00010000));
	BOOST_REQUIRE(f.contains(b00011000));
	BOOST_REQUIRE(f.contains(b00110000));
	BOOST_REQUIRE(f.contains(b00110010));
	BOOST_REQUIRE(f.contains(b00111000));
	BOOST_REQUIRE(f.contains(b00000110));
	BOOST_REQUIRE(f.contains(b00110110));
	BOOST_REQUIRE(!f.contains(b00110111));

	f.removeRaw(b00010000);
	BOOST_REQUIRE(!f.contains(b00000001));
	BOOST_REQUIRE(f.contains(b00010000));
	BOOST_REQUIRE(f.contains(b00011000));
	BOOST_REQUIRE(f.contains(b00110000));
	BOOST_REQUIRE(f.contains(b00110010));
	BOOST_REQUIRE(f.contains(b00111000));
	BOOST_REQUIRE(f.contains(b00000110));
	BOOST_REQUIRE(f.contains(b00110110));
	BOOST_REQUIRE(!f.contains(b00110111));

	f.removeRaw(b00111000);
	BOOST_REQUIRE(!f.contains(b00000001));
	BOOST_REQUIRE(f.contains(b00010000));
	BOOST_REQUIRE(!f.contains(b00011000));
	BOOST_REQUIRE(f.contains(b00110000));
	BOOST_REQUIRE(f.contains(b00110010));
	BOOST_REQUIRE(!f.contains(b00111000));
	BOOST_REQUIRE(f.contains(b00000110));
	BOOST_REQUIRE(f.contains(b00110110));
	BOOST_REQUIRE(!f.contains(b00110111));

	f.addRaw(b00000001);
	BOOST_REQUIRE(f.contains(b00000001));
	BOOST_REQUIRE(f.contains(b00010000));
	BOOST_REQUIRE(!f.contains(b00011000));
	BOOST_REQUIRE(f.contains(b00110000));
	BOOST_REQUIRE(f.contains(b00110010));
	BOOST_REQUIRE(!f.contains(b00111000));
	BOOST_REQUIRE(f.contains(b00000110));
	BOOST_REQUIRE(f.contains(b00110110));
	BOOST_REQUIRE(f.contains(b00110111));

	f.removeRaw(b00110111);
	BOOST_REQUIRE(!f.contains(b00000001));
	BOOST_REQUIRE(f.contains(b00010000));
	BOOST_REQUIRE(!f.contains(b00011000));
	BOOST_REQUIRE(!f.contains(b00110000));
	BOOST_REQUIRE(!f.contains(b00110010));
	BOOST_REQUIRE(!f.contains(b00111000));
	BOOST_REQUIRE(!f.contains(b00000110));
	BOOST_REQUIRE(!f.contains(b00110110));
	BOOST_REQUIRE(!f.contains(b00110111));

	f.removeRaw(b00110111);
	BOOST_REQUIRE(!f.contains(b00000001));
	BOOST_REQUIRE(!f.contains(b00010000));
	BOOST_REQUIRE(!f.contains(b00011000));
	BOOST_REQUIRE(!f.contains(b00110000));
	BOOST_REQUIRE(!f.contains(b00110010));
	BOOST_REQUIRE(!f.contains(b00111000));
	BOOST_REQUIRE(!f.contains(b00000110));
	BOOST_REQUIRE(!f.contains(b00110110));
	BOOST_REQUIRE(!f.contains(b00110111));
}

BOOST_AUTO_TEST_SUITE_END()