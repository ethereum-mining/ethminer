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
/** @file FixedHash.h
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 *
 * The FixedHash fixed-size "hash" container type.
 */

#pragma once

#include <string>
#include <libdevcore/FixedHash.h>
#include <libdevcore/vector_ref.h>

namespace dev
{

// SHA-3 convenience routines.

/// Calculate SHA3-256 hash of the given input and load it into the given output.
void sha3(bytesConstRef _input, bytesRef _output);

/// Calculate SHA3-256 hash of the given input, possibly interpreting it as nibbles, and return the hash as a string filled with binary data.
std::string sha3(std::string const& _input, bool _isNibbles);

/// Calculate SHA3-256 hash of the given input, returning as a byte array.
bytes sha3Bytes(bytesConstRef _input);

/// Calculate SHA3-256 hash of the given input (presented as a binary string), returning as a byte array.
inline bytes sha3Bytes(std::string const& _input) { return sha3Bytes((std::string*)&_input); }

/// Calculate SHA3-256 hash of the given input, returning as a byte array.
inline bytes sha3Bytes(bytes const& _input) { return sha3Bytes((bytes*)&_input); }

/// Calculate SHA3-256 hash of the given input, returning as a 256-bit hash.
h256 sha3(bytesConstRef _input);

/// Calculate SHA3-256 hash of the given input, returning as a 256-bit hash.
inline h256 sha3(bytes const& _input) { return sha3(bytesConstRef((bytes*)&_input)); }

/// Calculate SHA3-256 hash of the given input (presented as a binary-filled string), returning as a 256-bit hash.
inline h256 sha3(std::string const& _input) { return sha3(bytesConstRef(_input)); }
	
/// Calculate SHA3-256 MAC
void sha3mac(bytesConstRef _secret, bytesConstRef _plain, bytesRef _output);

/// Calculate SHA3-256 hash of the given input (presented as a FixedHash), returns a 256-bit hash.
template<unsigned N> inline h256 sha3(FixedHash<N> const& _input) { return sha3(_input.ref()); }

extern h256 EmptySHA3;

extern h256 EmptyListSHA3;

// Other crypto convenience routines

bytes aesDecrypt(bytesConstRef _cipher, std::string const& _password, unsigned _rounds = 2000, bytesConstRef _salt = bytesConstRef());

void sha256(bytesConstRef _input, bytesRef _output);

void ripemd160(bytesConstRef _input, bytesRef _output);

}
