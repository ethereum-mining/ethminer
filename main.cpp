#include "RLP.h"
#include "PatriciaTree.h"
#include "VirtualMachine.h"
using namespace std;
using namespace eth;

std::string asHex(std::string const& _data)
{
	std::ostringstream ret;
	for (auto i: _data)
		ret << hex << setfill('0') << setw(2) << (int)i;
	return ret.str();
}

int main()
{
	// int of value 15
	assert(toString(RLP("\x0f")) == "15");

	// 2-item list
	assert(toString(RLP("\x43""dog")) == "\"dog\"");

	// 3-character string
	assert(toString(RLP("\x82\x0f\x43""dog")) == "[ 15, \"dog\" ]");

	// 1-byte (8-bit) int
	assert(toString(RLP("\x18\x45")) == "69");

	// 2-byte (16-bit) int
	assert(toString(RLP("\x19\x01\x01")) == "257");

	// 32-byte (256-bit) int
	ostringstream o1;
	o1 << hex << RLP("\x37\x10\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f");
	assert(o1.str() == "100102030405060708090A0B0C0D0E0F101112131415161718191A1B1C1D1E1F");

	// 33-byte (264-bit) int
	ostringstream o2;
	o2 << hex << RLP("\x38\x21\x20\x10\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f");
	assert(o2.str() == "2120100102030405060708090A0B0C0D0E0F101112131415161718191A1B1C1D1E1F");

	// 56-character string.
	assert(toString(RLP("\x78\x38""Lorem ipsum dolor sit amet, consectetur adipisicing elit")) == "\"Lorem ipsum dolor sit amet, consectetur adipisicing elit\"");

	/*
	 * Hex-prefix Notation. First nibble has flags: oddness = 2^0 & termination = 2^1
	 * [0,0,1,2,3,4,5]   0x10012345
	 * [0,1,2,3,4,5]     0x00012345
	 * [1,2,3,4,5]       0x112345
	 * [0,0,1,2,3,4]     0x00001234
	 * [0,1,2,3,4]       0x101234
	 * [1,2,3,4]         0x001234
	 * [0,0,1,2,3,4,5,T] 0x30012345
	 * [0,0,1,2,3,4,T]   0x20001234
	 * [0,1,2,3,4,5,T]   0x20012345
	 * [1,2,3,4,5,T]     0x312345
	 * [1,2,3,4,T]       0x201234
	 */
	assert(asHex(fromHex({0, 0, 1, 2, 3, 4, 5}, false)) == "10012345");
	assert(asHex(fromHex({0, 1, 2, 3, 4, 5}, false)) == "00012345");
	assert(asHex(fromHex({1, 2, 3, 4, 5}, false)) == "112345");
	assert(asHex(fromHex({0, 0, 1, 2, 3, 4}, false)) == "00001234");
	assert(asHex(fromHex({0, 1, 2, 3, 4}, false)) == "101234");
	assert(asHex(fromHex({1, 2, 3, 4}, false)) == "001234");
	assert(asHex(fromHex({0, 0, 1, 2, 3, 4, 5}, true)) == "30012345");
	assert(asHex(fromHex({0, 0, 1, 2, 3, 4}, true)) == "20001234");
	assert(asHex(fromHex({0, 1, 2, 3, 4, 5}, true)) == "20012345");
	assert(asHex(fromHex({1, 2, 3, 4, 5}, true)) == "312345");
	assert(asHex(fromHex({1, 2, 3, 4}, true)) == "201234");

	return 0;
}

