/*
	This file is part of cpp-ethereum.

	cpp-ethereum is free software: you can redistribute it and/or modify
	it under the terms of the GNU General Public License as published by
	the Free Software Foundation, either version 3 of the License, or
	(at your option) any later version.

	cpp-ethereum is distributed in the hope that it will be useful,
	but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	GNU General Public License for more details.

	You should have received a copy of the GNU General Public License
	along with cpp-ethereum.  If not, see <http://www.gnu.org/licenses/>.
*/
/** @file CommonJS.h
 * @authors:
 *   Gav Wood <i@gavwood.com>
 * @date 2014
 */

#pragma once

#include <string>
#include <libethcore/Common.h>
#include <libethcore/CommonIO.h>
#include <libethcore/CommonData.h>
#include <libethcore/FixedHash.h>
#include <libethereum/CommonEth.h>

namespace eth
{

eth::bytes jsToBytes(std::string const& _s);
std::string jsPadded(std::string const& _s, unsigned _l, unsigned _r);
std::string jsPadded(std::string const& _s, unsigned _l);
std::string jsUnpadded(std::string _s);

template <unsigned N> eth::FixedHash<N> jsToFixed(std::string const& _s)
{
	if (_s.substr(0, 2) == "0x")
		// Hex
		return eth::FixedHash<N>(_s.substr(2));
	else if (_s.find_first_not_of("0123456789") == std::string::npos)
		// Decimal
		return (typename eth::FixedHash<N>::Arith)(_s);
	else
		// Binary
		return eth::FixedHash<N>(asBytes(jsPadded(_s, N)));
}

template <unsigned N> boost::multiprecision::number<boost::multiprecision::cpp_int_backend<N * 8, N * 8, boost::multiprecision::unsigned_magnitude, boost::multiprecision::unchecked, void>> jsToInt(std::string const& _s)
{
	if (_s.substr(0, 2) == "0x")
		// Hex
		return eth::fromBigEndian<boost::multiprecision::number<boost::multiprecision::cpp_int_backend<N * 8, N * 8, boost::multiprecision::unsigned_magnitude, boost::multiprecision::unchecked, void>>>(eth::fromHex(_s.substr(2)));
	else if (_s.find_first_not_of("0123456789") == std::string::npos)
		// Decimal
		return boost::multiprecision::number<boost::multiprecision::cpp_int_backend<N * 8, N * 8, boost::multiprecision::unsigned_magnitude, boost::multiprecision::unchecked, void>>(_s);
	else
		// Binary
		return eth::fromBigEndian<boost::multiprecision::number<boost::multiprecision::cpp_int_backend<N * 8, N * 8, boost::multiprecision::unsigned_magnitude, boost::multiprecision::unchecked, void>>>(asBytes(jsPadded(_s, N)));
}

inline eth::Address jsToAddress(std::string const& _s) { return jsToFixed<20>(_s); }
inline eth::Secret jsToSecret(std::string const& _s) { return jsToFixed<32>(_s); }
inline eth::u256 jsToU256(std::string const& _s) { return jsToInt<32>(_s); }

template <unsigned S> std::string toJS(eth::FixedHash<S> const& _h) { return "0x" + toHex(_h.ref()); }
template <unsigned N> std::string toJS(boost::multiprecision::number<boost::multiprecision::cpp_int_backend<N, N, boost::multiprecision::unsigned_magnitude, boost::multiprecision::unchecked, void>> const& _n) { return "0x" + eth::toHex(eth::toCompactBigEndian(_n)); }

inline std::string jsToBinary(std::string const& _s)
{
	return eth::asString(jsToBytes(_s));
}

inline std::string jsToDecimal(std::string const& _s)
{
	return eth::toString(jsToU256(_s));
}

inline std::string jsToHex(std::string const& _s)
{
	return "0x" + eth::toHex(asBytes(_s));
}

}
