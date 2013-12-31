#pragma once

#include <cassert>
#include <sstream>
#include <cstdint>
#include <boost/multiprecision/cpp_int.hpp>
#include "foreign.h"

namespace eth
{

using byte = uint8_t;
using bytes = std::vector<byte>;

using fBytes = foreign<byte>;
using fConstBytes = foreign<byte const>;

using bigint = boost::multiprecision::number<boost::multiprecision::cpp_int_backend<>>;
using u256 =  boost::multiprecision::number<boost::multiprecision::cpp_int_backend<256, 256, boost::multiprecision::unsigned_magnitude, boost::multiprecision::unchecked, void>>;
using s256 =  boost::multiprecision::number<boost::multiprecision::cpp_int_backend<256, 256, boost::multiprecision::signed_magnitude, boost::multiprecision::unchecked, void>>;
using uint = uint64_t;
using sint = int64_t;

template <class _T> std::string toString(_T const& _t) { std::ostringstream o; o << _t; return o.str(); }

template <class _T> inline std::string asHex(_T const& _data)
{
	std::ostringstream ret;
	for (auto i: _data)
		ret << std::hex << std::setfill('0') << std::setw(2) << (int)i;
	return ret.str();
}

}
