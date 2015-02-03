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

#include <libdevcore/Common.h>
#include <libdevcore/FixedHash.h>
#include <libdevcrypto/Common.h>

namespace dev
{
namespace eth
{

/// Current protocol version.
extern const unsigned c_protocolVersion;

/// Current database version.
extern const unsigned c_databaseVersion;

/// User-friendly string representation of the amount _b in wei.
std::string formatBalance(bigint const& _b);

/// Get information concerning the currency denominations.
std::vector<std::pair<u256, std::string>> const& units();

/// The log bloom's size (512 bit).
using LogBloom = h512;

// The various denominations; here for ease of use where needed within code.
/*static const u256 Uether = ((((u256(1000000000) * 1000000000) * 1000000000) * 1000000000) * 1000000000) * 1000000000;
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
static const u256 grand = (u256(1000000000) * 1000000000) * 1000;*/
static const u256 ether = u256(1000000000) * 1000000000;
static const u256 finney = u256(1000000000) * 1000000;
static const u256 szabo = u256(1000000000) * 1000;
/*static const u256 Gwei = u256(1000000000);
static const u256 Mwei = u256(1000000);
static const u256 Kwei = u256(1000);*/
static const u256 wei = u256(1);

}
}
