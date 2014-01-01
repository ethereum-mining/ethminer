#include "Common.h"
#include "RLP.h"
#include "PatriciaTree.h"
#include "VirtualMachine.h"
using namespace std;
using namespace eth;

template <class _T> void rlpListAux(RLPStream& _out, _T _t)
{
	_out << _t;
}

template <class _T, class ... _Ts> void rlpListAux(RLPStream& _out, _T _t, _Ts ... _ts)
{
	_out << _t;
	rlpListAux(_out, _ts...);
}

template <class _T> std::string rlp(_T _t)
{
	RLPStream out;
	out << _t;
	return out.str();
}

template <class ... _Ts> std::string rlpList(_Ts ... _ts)
{
	RLPStream out;
	out << RLPList(sizeof ...(_Ts));
	rlpListAux(out, _ts...);
	return out.str();
}

int main()
{
	cout << hex << hash256({{"dog", "puppy"}}) << endl;
	cout << hex << hash256({{"dog", "puppy"}, {"doe", "reindeer"}}) << endl;
	cout << hex << hash256({{"doe", "reindeer"}, {"dog", "puppy"}, {"dogglesworth", "cat"}}) << endl;
	cout << hex << hash256({{"dog", "puppy"}, {"horse", "stallion"}, {"do", "verb"}, {"doge", "coin"}}) << endl;

	// int of value 15
	assert(RLP("\x0f") == 15);
	assert(rlp(15) == "\x0f");

	// 3-character string
	assert(RLP("\x43""dog") == "dog");
	assert(rlp("dog") == "\x43""dog");

	// 2-item list
	RLP twoItemList("\x82\x0f\x43""dog");
	assert(twoItemList.itemCount() == 2 && twoItemList[0] == 15 && twoItemList[1] == "dog");
	assert(rlpList(15, "dog") == "\x82\x0f\x43""dog");

	// 1-byte (8-bit) int
	assert(RLP("\x18\x45") == 69);
	assert(rlp(69) == "\x18\x45");

	// 2-byte (16-bit) int
	assert(RLP("\x19\x01\x01") == 257);
	assert(rlp(257) == "\x19\x01\x01");

	// 32-byte (256-bit) int
	assert(RLP("\x37\x10\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f") == bigint("0x100102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f"));
	assert(rlp(bigint("0x100102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f")) == "\x37\x10\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f");

	// 33-byte (264-bit) int
	assert(RLP("\x38\x21\x20\x10\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f") == bigint("0x20100102030405060708090A0B0C0D0E0F101112131415161718191A1B1C1D1E1F"));
	assert(rlp(bigint("0x20100102030405060708090A0B0C0D0E0F101112131415161718191A1B1C1D1E1F")) == "\x38\x21\x20\x10\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f");

	// 56-character string.
	assert(RLP("\x78\x38""Lorem ipsum dolor sit amet, consectetur adipisicing elit") == "Lorem ipsum dolor sit amet, consectetur adipisicing elit");
	assert(rlp("Lorem ipsum dolor sit amet, consectetur adipisicing elit") == "\x78\x38""Lorem ipsum dolor sit amet, consectetur adipisicing elit");

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
	assert(asHex(hexPrefixEncode({0, 0, 1, 2, 3, 4, 5}, false)) == "10012345");
	assert(asHex(hexPrefixEncode({0, 1, 2, 3, 4, 5}, false)) == "00012345");
	assert(asHex(hexPrefixEncode({1, 2, 3, 4, 5}, false)) == "112345");
	assert(asHex(hexPrefixEncode({0, 0, 1, 2, 3, 4}, false)) == "00001234");
	assert(asHex(hexPrefixEncode({0, 1, 2, 3, 4}, false)) == "101234");
	assert(asHex(hexPrefixEncode({1, 2, 3, 4}, false)) == "001234");
	assert(asHex(hexPrefixEncode({0, 0, 1, 2, 3, 4, 5}, true)) == "30012345");
	assert(asHex(hexPrefixEncode({0, 0, 1, 2, 3, 4}, true)) == "20001234");
	assert(asHex(hexPrefixEncode({0, 1, 2, 3, 4, 5}, true)) == "20012345");
	assert(asHex(hexPrefixEncode({1, 2, 3, 4, 5}, true)) == "312345");
	assert(asHex(hexPrefixEncode({1, 2, 3, 4}, true)) == "201234");

	return 0;
}

