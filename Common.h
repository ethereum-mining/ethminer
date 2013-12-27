#pragma once

#include <sstream>
#include "foreign.h"

namespace eth
{

typedef uint8_t byte;
typedef foreign<byte> Bytes;

template <class _T> std::string toString(_T const& _t) { std::ostringstream o; o << _t; return o.str(); }

}
