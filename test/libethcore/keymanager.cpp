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
/** @file keymanager.cpp
 * @author Christoph Jentzsch <cj@ethdev.com>
 * @date 2015
 * Keymanager test functions.
 */


#include <boost/test/unit_test.hpp>
#include <test/TestHelper.h>
#include <libdevcore/TransientDirectory.h>
#include <libethcore/KeyManager.h>

using namespace std;
using namespace dev;
using namespace dev::eth;

BOOST_AUTO_TEST_SUITE(KeyManagerTests)

BOOST_AUTO_TEST_CASE(KeyInfoDefaultConstructor)
{
	cnote << "KeyInfoDefaultConstructor";
	KeyInfo kiDefault;
	BOOST_CHECK_EQUAL(kiDefault.accountName, "");
	BOOST_CHECK(kiDefault.passHash == h256());
}

BOOST_AUTO_TEST_CASE(KeyInfoConstructor)
{
	cnote << "KeyInfoConstructor";
	h256 passHash("0x2a");
	string accountName = "myAccount";
	KeyInfo ki(passHash, accountName);
	BOOST_CHECK_EQUAL(ki.accountName, "myAccount");
	BOOST_CHECK(ki.passHash == h256("0x2a"));
}

BOOST_AUTO_TEST_CASE(KeyManagerConstructor)
{
	cnote << "KeyManagerConstructor";
	KeyManager km;
	BOOST_CHECK_EQUAL(km.keysFile(), km.defaultPath());
	BOOST_CHECK_EQUAL(km.defaultPath(), getDataDir("ethereum") + "/keys.info");
	BOOST_CHECK(km.store().keys() == SecretStore().keys());
	for (auto a: km.accounts())
		km.kill(a);
}

BOOST_AUTO_TEST_CASE(KeyManagerKeysFile)
{
	cnote << "KeyManagerKeysFile";
	KeyManager km;
	string password = "hardPassword";
	BOOST_CHECK(!km.load(password));

	// set to valid path
	TransientDirectory tmpDir;
	km.setKeysFile(tmpDir.path());
	BOOST_CHECK(!km.exists());
	BOOST_CHECK_THROW(km.create(password), FileError);
	km.setKeysFile(tmpDir.path() + "/notExistingDir/keysFile.json");
	BOOST_CHECK_NO_THROW(km.create(password));
	BOOST_CHECK(km.exists());
	km.setKeysFile(tmpDir.path() + "keysFile.json");
	BOOST_CHECK_NO_THROW(km.create(password));
	km.save(password);
	BOOST_CHECK(km.load(password));

	for (auto a: km.accounts())
		km.kill(a);
}

BOOST_AUTO_TEST_CASE(KeyManagerHints)
{
	cnote << "KeyManagerHints";
	KeyManager km;
	string password = "hardPassword";

	// set to valid path
	TransientDirectory tmpDir;
	km.setKeysFile(tmpDir.path() + "keysFile.json");
	km.create(password);
	km.save(password);

	BOOST_CHECK(!km.haveHint(password + "2"));
	km.notePassword(password);
	BOOST_CHECK(km.haveHint(password));

	for (auto a: km.accounts())
		km.kill(a);
}

BOOST_AUTO_TEST_CASE(KeyManagerAccounts)
{
	string password = "hardPassword";

	TransientDirectory tmpDir;
	KeyManager km(tmpDir.path()+ "keysFile.json", tmpDir.path());

	BOOST_CHECK_NO_THROW(km.create(password));
	BOOST_CHECK(km.accounts().empty());
	BOOST_CHECK(km.load(password));

	for (auto a: km.accounts())
		km.kill(a);
}

BOOST_AUTO_TEST_CASE(KeyManagerKill)
{
	string password = "hardPassword";
	TransientDirectory tmpDir;
	KeyPair kp = KeyPair::create();

	{
		KeyManager km(tmpDir.path() + "keysFile.json", tmpDir.path());
		BOOST_CHECK_NO_THROW(km.create(password));
		BOOST_CHECK(km.accounts().empty());
		BOOST_CHECK(km.load(password));
		BOOST_CHECK(km.import(kp.secret(), "test"));
	}
	{
		KeyManager km(tmpDir.path() + "keysFile.json", tmpDir.path());
		BOOST_CHECK(km.load(password));
		Addresses addresses = km.accounts();
		BOOST_CHECK(addresses.size() == 1 && addresses[0] == kp.address());
		km.kill(addresses[0]);
	}
	{
		KeyManager km(tmpDir.path() + "keysFile.json", tmpDir.path());
		BOOST_CHECK(km.load(password));
		BOOST_CHECK(km.accounts().empty());
	}
}

BOOST_AUTO_TEST_SUITE_END()
