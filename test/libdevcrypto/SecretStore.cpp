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
/** @file SecretStore.cpp
 * @author Gav Wood <i@gavwood.com>
 * @date 2015
 * Secret store test functions.
 */

#include <fstream>
#include <random>
#include <boost/test/unit_test.hpp>
#include "../JsonSpiritHeaders.h"
#include <libdevcrypto/SecretStore.h>
#include <libdevcore/CommonIO.h>
#include <libdevcore/TrieDB.h>
#include <libdevcore/TrieHash.h>
#include "MemTrie.h"
#include <test/TestHelper.h>
#include <test/TestUtils.h>
using namespace std;
using namespace dev;
using namespace dev::test;

namespace js = json_spirit;
namespace fs = boost::filesystem;

BOOST_GLOBAL_FIXTURE( MoveNonceToTempDir );

BOOST_AUTO_TEST_SUITE(KeyStore)

BOOST_AUTO_TEST_CASE(basic_tests)
{
	string testPath = test::getTestPath();

	testPath += "/KeyStoreTests";

	cnote << "Testing Key Store...";
	js::mValue v;
	string s = contentsString(testPath + "/basic_tests.json");
	BOOST_REQUIRE_MESSAGE(s.length() > 0, "Contents of 'KeyStoreTests/basic_tests.json' is empty. Have you cloned the 'tests' repo branch develop?");
	js::read_string(s, v);
	for (auto& i: v.get_obj())
	{
		cnote << i.first;
		js::mObject& o = i.second.get_obj();
		TransientDirectory tmpDir;
		SecretStore store(tmpDir.path());
		h128 u = store.readKeyContent(js::write_string(o["json"], false));
		cdebug << "read uuid" << u;
		bytes s = store.secret(u, [&](){ return o["password"].get_str(); });
		cdebug << "got secret" << toHex(s);
		BOOST_REQUIRE_EQUAL(toHex(s), o["priv"].get_str());
	}
}

BOOST_AUTO_TEST_CASE(import_key_from_file)
{
	// Imports a key from an external file. Tests that the imported key is there
	// and that the external file is not deleted.
	TransientDirectory importDir;
	string importFile = importDir.path() + "/import.json";
	TransientDirectory storeDir;
	string keyData = R"({
		"version": 3,
		"crypto": {
			"ciphertext": "d69313b6470ac1942f75d72ebf8818a0d484ac78478a132ee081cd954d6bd7a9",
			"cipherparams": { "iv": "ffffffffffffffffffffffffffffffff" },
			"kdf": "pbkdf2",
			"kdfparams": { "dklen": 32,  "c": 262144,  "prf": "hmac-sha256",  "salt": "c82ef14476014cbf438081a42709e2ed" },
			"mac": "cf6bfbcc77142a22c4a908784b4a16f1023a1d0e2aff404c20158fa4f1587177",
			"cipher": "aes-128-ctr",
			"version": 1
		},
		"id": "abb67040-8dbe-0dad-fc39-2b082ef0ee5f"
	})";
	string password = "bar";
	string priv = "0202020202020202020202020202020202020202020202020202020202020202";
	writeFile(importFile, keyData);

	h128 uuid;
	{
		SecretStore store(storeDir.path());
		BOOST_CHECK_EQUAL(store.keys().size(), 0);
		uuid = store.importKey(importFile);
		BOOST_CHECK(!!uuid);
		BOOST_CHECK(contentsString(importFile) == keyData);
		BOOST_CHECK_EQUAL(priv, toHex(store.secret(uuid, [&](){ return password; })));
		BOOST_CHECK_EQUAL(store.keys().size(), 1);
	}
	fs::remove(importFile);
	// now do it again to check whether SecretStore properly stored it on disk
	{
		SecretStore store(storeDir.path());
		BOOST_CHECK_EQUAL(store.keys().size(), 1);
		BOOST_CHECK_EQUAL(priv, toHex(store.secret(uuid, [&](){ return password; })));
	}
}

BOOST_AUTO_TEST_CASE(import_secret)
{
	for (string const& password: {"foobar", ""})
	{
		TransientDirectory storeDir;
		string priv = "0202020202020202020202020202020202020202020202020202020202020202";

		h128 uuid;
		{
			SecretStore store(storeDir.path());
			BOOST_CHECK_EQUAL(store.keys().size(), 0);
			uuid = store.importSecret(fromHex(priv), password);
			BOOST_CHECK(!!uuid);
			BOOST_CHECK_EQUAL(priv, toHex(store.secret(uuid, [&](){ return password; })));
			BOOST_CHECK_EQUAL(store.keys().size(), 1);
		}
		{
			SecretStore store(storeDir.path());
			BOOST_CHECK_EQUAL(store.keys().size(), 1);
			BOOST_CHECK_EQUAL(priv, toHex(store.secret(uuid, [&](){ return password; })));
		}
	}
}

BOOST_AUTO_TEST_CASE(wrong_password)
{
	TransientDirectory storeDir;
	SecretStore store(storeDir.path());
	string password = "foobar";
	string priv = "0202020202020202020202020202020202020202020202020202020202020202";

	h128 uuid;
	{
		SecretStore store(storeDir.path());
		BOOST_CHECK_EQUAL(store.keys().size(), 0);
		uuid = store.importSecret(fromHex(priv), password);
		BOOST_CHECK(!!uuid);
		BOOST_CHECK_EQUAL(priv, toHex(store.secret(uuid, [&](){ return password; })));
		BOOST_CHECK_EQUAL(store.keys().size(), 1);
		// password will not be queried
		BOOST_CHECK_EQUAL(priv, toHex(store.secret(uuid, [&](){ return "abcdefg"; })));
	}
	{
		SecretStore store(storeDir.path());
		BOOST_CHECK_EQUAL(store.keys().size(), 1);
		BOOST_CHECK(store.secret(uuid, [&](){ return "abcdefg"; }).empty());
	}
}

BOOST_AUTO_TEST_CASE(recode)
{
	TransientDirectory storeDir;
	SecretStore store(storeDir.path());
	string password = "foobar";
	string changedPassword = "abcdefg";
	string priv = "0202020202020202020202020202020202020202020202020202020202020202";

	h128 uuid;
	{
		SecretStore store(storeDir.path());
		BOOST_CHECK_EQUAL(store.keys().size(), 0);
		uuid = store.importSecret(fromHex(priv), password);
		BOOST_CHECK(!!uuid);
		BOOST_CHECK_EQUAL(priv, toHex(store.secret(uuid, [&](){ return password; })));
		BOOST_CHECK_EQUAL(store.keys().size(), 1);
	}
	{
		SecretStore store(storeDir.path());
		BOOST_CHECK_EQUAL(store.keys().size(), 1);
		BOOST_CHECK(store.secret(uuid, [&](){ return "abcdefg"; }).empty());
		BOOST_CHECK(store.recode(uuid, changedPassword, [&](){ return password; }));
		BOOST_CHECK_EQUAL(store.keys().size(), 1);
		BOOST_CHECK_EQUAL(priv, toHex(store.secret(uuid, [&](){ return changedPassword; })));
		store.clearCache();
		BOOST_CHECK(store.secret(uuid, [&](){ return password; }).empty());
		BOOST_CHECK_EQUAL(priv, toHex(store.secret(uuid, [&](){ return changedPassword; })));
	}
	{
		SecretStore store(storeDir.path());
		BOOST_CHECK_EQUAL(store.keys().size(), 1);
		BOOST_CHECK(store.secret(uuid, [&](){ return password; }).empty());
		BOOST_CHECK_EQUAL(priv, toHex(store.secret(uuid, [&](){ return changedPassword; })));
	}
}

BOOST_AUTO_TEST_SUITE_END()
