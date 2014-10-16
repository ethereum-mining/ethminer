#include "CommonJS.h"

namespace dev {
namespace eth {

bytes dev::eth::jsToBytes(std::string const& _s)
{
	if (_s.substr(0, 2) == "0x")
		// Hex
		return fromHex(_s.substr(2));
	else if (_s.find_first_not_of("0123456789") == std::string::npos)
		// Decimal
		return toCompactBigEndian(bigint(_s));
	else
		// Binary
		return asBytes(_s);
}

std::string dev::eth::jsPadded(std::string const& _s, unsigned _l, unsigned _r)
{
	bytes b = jsToBytes(_s);
	while (b.size() < _l)
		b.insert(b.begin(), 0);
	while (b.size() < _r)
		b.push_back(0);
	return asString(b).substr(b.size() - std::max(_l, _r));
}

std::string dev::eth::jsPadded(std::string const& _s, unsigned _l)
{
	if (_s.substr(0, 2) == "0x" || _s.find_first_not_of("0123456789") == std::string::npos)
        // Numeric: pad to right
		return jsPadded(_s, _l, _l);
	else
        // Text: pad to the left
		return jsPadded(_s, 0, _l);
}

std::string dev::eth::jsUnpadded(std::string _s)
{
	auto p = _s.find_last_not_of((char)0);
	_s.resize(p == std::string::npos ? 0 : (p + 1));
	return _s;
}




}
}
