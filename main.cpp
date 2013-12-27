#include "RLP.h"
using namespace std;
using namespace eth;

int main()
{
	{
		string t = "\x0f";
		assert(toString(RLP(Bytes((byte*)t.data(), t.size()))) == "15");
	}
	{
		string t = "\x43""dog";
		assert(toString(RLP(Bytes((byte*)t.data(), t.size()))) == "\"dog\"");
	}
	{
		string t = "\x82\x0f\x43""dog";
		assert(toString(RLP(Bytes((byte*)t.data(), t.size()))) == "[ 15, \"dog\" ]");
	}
	{
		string t = "\x18\x45";
		assert(toString(RLP(Bytes((byte*)t.data(), t.size()))) == "69");
	}
	{
		string t = "\x19\x01\x01";
		assert(toString(RLP(Bytes((byte*)t.data(), t.size()))) == "257");
	}
	{
		string t = "\x37\x10\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f";
		ostringstream o;
		o << hex << RLP(Bytes((byte*)t.data(), t.size()));
		assert(o.str() == "100102030405060708090A0B0C0D0E0F101112131415161718191A1B1C1D1E1F");
	}
	{
		// 33-byte int
		string t = "\x38\x21\x20\x10\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f";
		ostringstream o;
		o << hex << RLP(Bytes((byte*)t.data(), t.size()));
		assert(o.str() == "20100102030405060708090A0B0C0D0E0F101112131415161718191A1B1C1D1E1F");
	}
	{
		string t = "\x78\x38""Lorem ipsum dolor sit amet, consectetur adipisicing elit";
		assert(toString(RLP(Bytes((byte*)t.data(), t.size()))) == "\"Lorem ipsum dolor sit amet, consectetur adipisicing elit\"");
	}
	return 0;
}

