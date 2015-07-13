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

using TopicBloomFilterShort = TopicBloomFilterBase<4>;
using TopicBloomFilterTest = TopicBloomFilterBase<TopicBloomFilterSize>;

void testAddNonExisting(TopicBloomFilterShort& _f, AbridgedTopic const& _h)
{
	BOOST_REQUIRE(!_f.containsRaw(_h));
	_f.addRaw(_h);
	BOOST_REQUIRE(_f.containsRaw(_h));
}

void testRemoveExisting(TopicBloomFilterShort& _f, AbridgedTopic const& _h)
{
	BOOST_REQUIRE(_f.containsRaw(_h));
	_f.removeRaw(_h);
	BOOST_REQUIRE(!_f.containsRaw(_h));
}

void testAddNonExistingBloom(TopicBloomFilterShort& _f, AbridgedTopic const& _h)
{
	BOOST_REQUIRE(!_f.containsBloom(_h));
	_f.addBloom(_h);
	BOOST_REQUIRE(_f.containsBloom(_h));
}

void testRemoveExistingBloom(TopicBloomFilterShort& _f, AbridgedTopic const& _h)
{
	BOOST_REQUIRE(_f.containsBloom(_h));
	_f.removeBloom(_h);
	BOOST_REQUIRE(!_f.containsBloom(_h));
}

double calculateExpected(TopicBloomFilterTest const& f, int n)
{
	int const m = f.size * 8; // number of bits in the bloom
	int const k = BitsPerBloom; // number of hash functions (e.g. bits set to 1 in every bloom)

	double singleBitSet = 1.0 / m; // probability of any bit being set after inserting a single bit
	double singleBitNotSet = (1.0 - singleBitSet);

	double singleNot = 1; // single bit not set after inserting N elements in the bloom filter
	for (int i = 0; i < k * n; ++i)
		singleNot *= singleBitNotSet;

	double single = 1.0 - singleNot; // probability of a single bit being set after inserting N elements in the bloom filter

	double kBitsSet = 1; // probability of K bits being set after inserting N elements in the bloom filter
	for (int i = 0; i < k; ++i)
		kBitsSet *= single;

	return kBitsSet;
}

double testFalsePositiveRate(TopicBloomFilterTest const& f, int inserted, Topic& x)
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

	double res = double(falsePositive) / double(c_sampleSize);

	double expected = calculateExpected(f, inserted);
	double allowed = expected * 1.2 + 0.05; // allow deviations ~25%

	//cnote << "Inserted: " << inserted << ", False Positive Rate: " << res << ", Expected: " << expected;
	BOOST_REQUIRE(res <= allowed);
	return expected;
}

BOOST_AUTO_TEST_SUITE(bloomFilter)

BOOST_AUTO_TEST_CASE(falsePositiveRate)
{
	VerbosityHolder setTemporaryLevel(10);
	cnote << "Testing Bloom Filter False Positive Rate...";

	TopicBloomFilterTest f;
	Topic x(0xC0DEFEED); // deterministic pseudorandom value

	double expectedRate = 0;

	for (int i = 1; i < 50 && isless(expectedRate, 0.5); ++i)
	{
		x = sha3(x);
		f.addBloom(AbridgedTopic(x));
		expectedRate = testFalsePositiveRate(f, i, x);
	}
}

BOOST_AUTO_TEST_CASE(bloomFilterRandom)
{
	VerbosityHolder setTemporaryLevel(10);
	cnote << "Testing Bloom Filter matching...";

	TopicBloomFilterShort f;
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

	TopicBloomFilterShort f;

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

static const unsigned DistributionTestSize = TopicBloomFilterSize;
static const unsigned TestArrSize = 8 * DistributionTestSize;

void updateDistribution(FixedHash<DistributionTestSize> const& _h, array<unsigned, TestArrSize>& _distribution)
{
	unsigned bits = 0;
	for (unsigned i = 0; i < DistributionTestSize; ++i)
		if (_h[i])
			for (unsigned j = 0; j < 8; ++j)
				if (_h[i] & c_powerOfTwoBitMmask[j])
				{
					_distribution[i * 8 + j]++;
					if (++bits >= BitsPerBloom)
						return;
				}
}

BOOST_AUTO_TEST_CASE(distributionRate)
{
	cnote << "Testing Bloom Filter Distribution Rate...";

	array<unsigned, TestArrSize> distribution;
	for (unsigned i = 0; i < TestArrSize; ++i)
		distribution[i] = 0;

	Topic x(0xC0FFEE); // deterministic pseudorandom value

	for (unsigned i = 0; i < 26000; ++i)
	{
		x = sha3(x);
		FixedHash<DistributionTestSize> h = TopicBloomFilter::bloom(abridge(x));
		updateDistribution(h, distribution);
	}

	unsigned average = 0;
	for (unsigned i = 0; i < TestArrSize; ++i)
		average += distribution[i];

	average /= TestArrSize;
	unsigned deviation = average / 3;
	unsigned maxAllowed = average + deviation;
	unsigned minAllowed = average - deviation;

	unsigned maximum = 0;
	unsigned minimum = 0xFFFFFFFF;

	for (unsigned i = 0; i < TestArrSize; ++i)
	{
		unsigned const& z = distribution[i];
		if (z > maximum)
			maximum = z;
		else if (z < minimum)
			minimum = z;
	}

	cnote << minimum << average << maximum;
	BOOST_REQUIRE(minimum > minAllowed);
	BOOST_REQUIRE(maximum < maxAllowed);
}

BOOST_AUTO_TEST_SUITE_END()
