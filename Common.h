#pragma once

#include <cassert>
#include <sstream>
#include <cstdint>
#include <boost/multiprecision/cpp_int.hpp>
#include "foreign.h"

namespace eth
{

using byte = uint8_t;
using Bytes = foreign<byte>;
using ConstBytes = foreign<byte const>;

using bigint = boost::multiprecision::cpp_int;
using uint = uint64_t;
using sint = int64_t;

template <class _T> std::string toString(_T const& _t) { std::ostringstream o; o << _t; return o.str(); }

}
