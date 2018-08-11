// ethminer -- Ethereum miner with OpenCL, CUDA and stratum support.
// Copyright 2018 ethminer Authors.
// Licensed under GNU General Public License, Version 3. See the LICENSE file.

/// @file
/// Very common stuff (i.e. that every other header needs except vector_ref.h).

#pragma once

#include "vector_ref.h"

#include <string>
#include <vector>

#include <boost/multiprecision/cpp_int.hpp>

using byte = uint8_t;

namespace dev
{
// Binary data types.
using bytes = std::vector<byte>;
using bytesRef = vector_ref<byte>;
using bytesConstRef = vector_ref<byte const>;

// Numeric types.
using bigint = boost::multiprecision::number<boost::multiprecision::cpp_int_backend<>>;
using u64 = boost::multiprecision::number<boost::multiprecision::cpp_int_backend<64, 64,
    boost::multiprecision::unsigned_magnitude, boost::multiprecision::unchecked, void>>;
using u128 = boost::multiprecision::number<boost::multiprecision::cpp_int_backend<128, 128,
    boost::multiprecision::unsigned_magnitude, boost::multiprecision::unchecked, void>>;
using u256 = boost::multiprecision::number<boost::multiprecision::cpp_int_backend<256, 256,
    boost::multiprecision::unsigned_magnitude, boost::multiprecision::unchecked, void>>;
using u160 = boost::multiprecision::number<boost::multiprecision::cpp_int_backend<160, 160,
    boost::multiprecision::unsigned_magnitude, boost::multiprecision::unchecked, void>>;
using u512 = boost::multiprecision::number<boost::multiprecision::cpp_int_backend<512, 512,
    boost::multiprecision::unsigned_magnitude, boost::multiprecision::unchecked, void>>;

// Null/Invalid values for convenience.
static const u256 Invalid256 = ~(u256)0;

/// Converts arbitrary value to string representation using std::stringstream.
template <class _T>
std::string toString(_T const& _t)
{
    std::ostringstream o;
    o << _t;
    return o.str();
}

}  // namespace dev
