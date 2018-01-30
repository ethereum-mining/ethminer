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
/** @file SHA3.h
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 *
 * The FixedHash fixed-size "hash" container type.
 */

#pragma once

#include <string>
#include "FixedHash.h"
#include "vector_ref.h"

namespace dev
{

// SHA-3 convenience routines.

/// Calculate SHA3-256 hash of the given input and load it into the given output.
/// @returns false if o_output.size() != 32.
bool sha3(bytesConstRef _input, bytesRef o_output);

/// Calculate SHA3-256 hash of the given input, returning as a 256-bit hash.
inline h256 sha3(bytesConstRef _input) { h256 ret; sha3(_input, ret.ref()); return ret; }

/// Calculate SHA3-256 hash of the given input, returning as a 256-bit hash.
inline h256 sha3(bytes const& _input) { return sha3(bytesConstRef(&_input)); }

/// Calculate SHA3-256 hash of the given input (presented as a FixedHash), returns a 256-bit hash.
template<unsigned N> inline h256 sha3(FixedHash<N> const& _input) { return sha3(_input.ref()); }

}
