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
/** @file CommonEth.cpp
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 */

#include "CommonEth.h"
#include <random>
#include <secp256k1/secp256k1.h>
#include <libdevcrypto/SHA3.h>
#include "Exceptions.h"
using namespace std;
using namespace dev;
using namespace dev::eth;

//#define ETH_ADDRESS_DEBUG 1
namespace dev
{
namespace eth
{

const unsigned c_protocolVersion = 36;
const unsigned c_databaseVersion = 3;

static const vector<pair<u256, string>> g_units =
{
	{((((u256(1000000000) * 1000000000) * 1000000000) * 1000000000) * 1000000000) * 1000000000, "Uether"},
	{((((u256(1000000000) * 1000000000) * 1000000000) * 1000000000) * 1000000000) * 1000000, "Vether"},
	{((((u256(1000000000) * 1000000000) * 1000000000) * 1000000000) * 1000000000) * 1000, "Dether"},
	{(((u256(1000000000) * 1000000000) * 1000000000) * 1000000000) * 1000000000, "Nether"},
	{(((u256(1000000000) * 1000000000) * 1000000000) * 1000000000) * 1000000, "Yether"},
	{(((u256(1000000000) * 1000000000) * 1000000000) * 1000000000) * 1000, "Zether"},
	{((u256(1000000000) * 1000000000) * 1000000000) * 1000000000, "Eether"},
	{((u256(1000000000) * 1000000000) * 1000000000) * 1000000, "Pether"},
	{((u256(1000000000) * 1000000000) * 1000000000) * 1000, "Tether"},
	{(u256(1000000000) * 1000000000) * 1000000000, "Gether"},
	{(u256(1000000000) * 1000000000) * 1000000, "Mether"},
	{(u256(1000000000) * 1000000000) * 1000, "Kether"},
	{u256(1000000000) * 1000000000, "ether"},
	{u256(1000000000) * 1000000, "finney"},
	{u256(1000000000) * 1000, "szabo"},
	{u256(1000000000), "Gwei"},
	{u256(1000000), "Mwei"},
	{u256(1000), "Kwei"},
	{u256(1), "wei"}
};

vector<pair<u256, string>> const& units()
{
	return g_units;
}

std::string formatBalance(u256 _b)
{
	ostringstream ret;
	if (_b > g_units[0].first * 10000)
	{
		ret << (_b / g_units[0].first) << " " << g_units[0].second;
		return ret.str();
	}
	ret << setprecision(5);
	for (auto const& i: g_units)
		if (i.first != 1 && _b >= i.first * 100)
		{
			ret << (double(_b / (i.first / 1000)) / 1000.0) << " " << i.second;
			return ret.str();
		}
	ret << _b << " wei";
	return ret.str();
}

Address toAddress(Secret _private)
{
	secp256k1_start();

	byte pubkey[65];
	int pubkeylen = 65;
	int ok = secp256k1_ecdsa_seckey_verify(_private.data());
	if (!ok)
		return Address();
	ok = secp256k1_ecdsa_pubkey_create(pubkey, &pubkeylen, _private.data(), 0);
	assert(pubkeylen == 65);
	if (!ok)
		return Address();
	ok = secp256k1_ecdsa_pubkey_verify(pubkey, 65);
	if (!ok)
		return Address();
	auto ret = right160(dev::sha3(bytesConstRef(&(pubkey[1]), 64)));
#if ETH_ADDRESS_DEBUG
	cout << "---- ADDRESS -------------------------------" << endl;
	cout << "SEC: " << _private << endl;
	cout << "PUB: " << toHex(bytesConstRef(&(pubkey[1]), 64)) << endl;
	cout << "ADR: " << ret << endl;
#endif
	return ret;
}

}}
