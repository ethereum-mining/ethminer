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
/** @file CommonEth.h
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 *
 * Ethereum-specific data structures & algorithms.
 */

#pragma once

#include "Common.h"
#include "FixedHash.h"

namespace eth
{

/// A secret key: 32 bytes.
/// @NOTE This is not endian-specific; it's just a bunch of bytes.
using Secret = h256;

/// A public key: 64 bytes.
/// @NOTE This is not endian-specific; it's just a bunch of bytes.
using Public = h512;

/// An Ethereum address: 20 bytes.
/// @NOTE This is not endian-specific; it's just a bunch of bytes.
using Address = h160;

/// A vector of Ethereum addresses.
using Addresses = h160s;

/// User-friendly string representation of the amount _b in wei.
std::string formatBalance(u256 _b);

/// Get information concerning the currency denominations.
std::vector<std::pair<u256, std::string>> const& units();

// The various denominations; here for ease of use where needed within code.
static const u256 Uether = ((((u256(1000000000) * 1000000000) * 1000000000) * 1000000000) * 1000000000) * 1000000000;
static const u256 Vether = ((((u256(1000000000) * 1000000000) * 1000000000) * 1000000000) * 1000000000) * 1000000;
static const u256 Dether = ((((u256(1000000000) * 1000000000) * 1000000000) * 1000000000) * 1000000000) * 1000;
static const u256 Nether = (((u256(1000000000) * 1000000000) * 1000000000) * 1000000000) * 1000000000;
static const u256 Yether = (((u256(1000000000) * 1000000000) * 1000000000) * 1000000000) * 1000000;
static const u256 Zether = (((u256(1000000000) * 1000000000) * 1000000000) * 1000000000) * 1000;
static const u256 Eether = ((u256(1000000000) * 1000000000) * 1000000000) * 1000000000;
static const u256 Pether = ((u256(1000000000) * 1000000000) * 1000000000) * 1000000;
static const u256 Tether = ((u256(1000000000) * 1000000000) * 1000000000) * 1000;
static const u256 Gether = (u256(1000000000) * 1000000000) * 1000000000;
static const u256 Mether = (u256(1000000000) * 1000000000) * 1000000;
static const u256 Kether = (u256(1000000000) * 1000000000) * 1000;
static const u256 ether = u256(1000000000) * 1000000000;
static const u256 finney = u256(1000000000) * 1000000;
static const u256 szabo = u256(1000000000) * 1000;
static const u256 Gwei = u256(1000000000);
static const u256 Mwei = u256(1000000);
static const u256 Kwei = u256(1000);
static const u256 wei = u256(1);

/// Convert a private key into the public key equivalent.
/// @returns 0 if it's not a valid private key.
Address toAddress(h256 _private);

/// Simple class that represents a "key pair".
/// All of the data of the class can be regenerated from the secret key (m_secret) alone.
/// Actually stores a tuplet of secret, public and address (the right 160-bits of the public).
class KeyPair
{
public:
	/// Null constructor.
	KeyPair() {}

	/// Normal constructor - populates object from the given secret key.
	KeyPair(Secret _k);

	/// Create a new, randomly generated object.
	static KeyPair create();

	/// Retrieve the secret key.
	Secret const& secret() const { return m_secret; }
	/// Retrieve the secret key.
	Secret const& sec() const { return m_secret; }

	/// Retrieve the public key.
	Public const& pub() const { return m_public; }

	/// Retrieve the associated address of the public key.
	Address const& address() const { return m_address; }

private:
	Secret m_secret;
	Public m_public;
	Address m_address;
};


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

}
