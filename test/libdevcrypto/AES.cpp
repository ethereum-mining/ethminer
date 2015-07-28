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
/** @file AES.cpp
 * @author Christoph Jentzsch <cj@ethdev.com>
 * @date 2015
 */

#include <boost/test/unit_test.hpp>
#include <libdevcore/Common.h>
#include <libdevcrypto/Common.h>
#include <libdevcore/SHA3.h>
#include <libdevcrypto/AES.h>
#include <libdevcore/FixedHash.h>

using namespace std;
using namespace dev;

BOOST_AUTO_TEST_SUITE(AES)

BOOST_AUTO_TEST_CASE(AesDecrypt)
{
	cout << "AesDecrypt" << endl;
	bytes seed = fromHex("2dbaead416c20cfd00c2fc9f1788ff9f965a2000799c96a624767cb0e1e90d2d7191efdd92349226742fdc73d1d87e2d597536c4641098b9a89836c94f58a2ab4c525c27c4cb848b3e22ea245b2bc5c8c7beaa900b0c479253fc96fce7ffc621");
	KeyPair kp(sha3Secure(aesDecrypt(&seed, "test")));
	BOOST_CHECK(Address("07746f871de684297923f933279555dda418f8a2") == kp.address());
}

BOOST_AUTO_TEST_CASE(AesDecryptWrongSeed)
{
	cout << "AesDecryptWrongSeed" << endl;
	bytes seed = fromHex("badaead416c20cfd00c2fc9f1788ff9f965a2000799c96a624767cb0e1e90d2d7191efdd92349226742fdc73d1d87e2d597536c4641098b9a89836c94f58a2ab4c525c27c4cb848b3e22ea245b2bc5c8c7beaa900b0c479253fc96fce7ffc621");
	KeyPair kp(sha3Secure(aesDecrypt(&seed, "test")));
	BOOST_CHECK(Address("07746f871de684297923f933279555dda418f8a2") != kp.address());
}

BOOST_AUTO_TEST_CASE(AesDecryptWrongPassword)
{
	cout << "AesDecryptWrongPassword" << endl;
	bytes seed = fromHex("2dbaead416c20cfd00c2fc9f1788ff9f965a2000799c96a624767cb0e1e90d2d7191efdd92349226742fdc73d1d87e2d597536c4641098b9a89836c94f58a2ab4c525c27c4cb848b3e22ea245b2bc5c8c7beaa900b0c479253fc96fce7ffc621");
	KeyPair kp(sha3Secure(aesDecrypt(&seed, "badtest")));
	BOOST_CHECK(Address("07746f871de684297923f933279555dda418f8a2") != kp.address());
}

BOOST_AUTO_TEST_CASE(AesDecryptFailInvalidSeed)
{
	cout << "AesDecryptFailInvalidSeed" << endl;
	bytes seed = fromHex("xdbaead416c20cfd00c2fc9f1788ff9f965a2000799c96a624767cb0e1e90d2d7191efdd92349226742fdc73d1d87e2d597536c4641098b9a89836c94f58a2ab4c525c27c4cb848b3e22ea245b2bc5c8c7beaa900b0c479253fc96fce7ffc621");
	BOOST_CHECK(bytes() == aesDecrypt(&seed, "test"));
}

BOOST_AUTO_TEST_CASE(AesDecryptFailInvalidSeedSize)
{
	cout << "AesDecryptFailInvalidSeedSize" << endl;
	bytes seed = fromHex("000102030405060708090a0b0c0d0e0f");
	BOOST_CHECK(bytes() == aesDecrypt(&seed, "test"));
}

BOOST_AUTO_TEST_CASE(AesDecryptFailInvalidSeed2)
{
	cout << "AesDecryptFailInvalidSeed2" << endl;
	bytes seed = fromHex("000102030405060708090a0b0c0d0e0f000102030405060708090a0b0c0d0e0f");
	BOOST_CHECK(bytes() == aesDecrypt(&seed, "test"));
}
BOOST_AUTO_TEST_CASE(AuthenticatedStreamConstructor)
{
	cout << "AuthenticatedStreamConstructor" << endl;

	Secret const sec(dev::sha3("test"));
	crypto::aes::AuthenticatedStream as(crypto::aes::Encrypt, sec, 0);
	BOOST_CHECK(as.getMacInterval() == 0);
	as.adjustInterval(1);
	BOOST_CHECK(as.getMacInterval() == 1);
	crypto::aes::AuthenticatedStream as_mac(crypto::aes::Encrypt, h128(), h128(), 42);
	BOOST_CHECK(as_mac.getMacInterval() == 42);
}

BOOST_AUTO_TEST_SUITE_END()

