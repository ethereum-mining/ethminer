#pragma once

#include <cassert>
#include <sstream>
#include <cstdint>
#include <boost/multiprecision/cpp_int.hpp>
#include "foreign.h"
#include "uint256_t.h"

namespace eth
{

using byte = uint8_t;
using bytes = vector<byte>;

using fBytes = foreign<byte>;
using fConstBytes = foreign<byte const>;

using bigint = boost::multiprecision::cpp_int;
using u256 = uint256_t;
using uint = uint64_t;
using sint = int64_t;

template <class _T> std::string toString(_T const& _t) { std::ostringstream o; o << _t; return o.str(); }

}
