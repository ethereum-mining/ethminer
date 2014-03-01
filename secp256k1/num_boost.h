// Copyright (c) 2013 Pieter Wuille
// Distributed under the MIT/X11 software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef _SECP256K1_NUM_REPR_
#define _SECP256K1_NUM_REPR_

#include <boost/multiprecision/cpp_int.hpp>

typedef boost::multiprecision::number<boost::multiprecision::cpp_int_backend<512, 512, boost::multiprecision::signed_magnitude, boost::multiprecision::unchecked, void>> secp256k1_num_t ;

#endif
