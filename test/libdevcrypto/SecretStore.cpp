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
#include "../TestHelper.h"
using namespace std;
using namespace dev;

namespace js = json_spirit;

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
		SecretStore store(".");
		h128 u = store.readKeyContent(js::write_string(o["json"], false));
		cdebug << "read uuid" << u;
		bytes s = store.secret(u, [&](){ return o["password"].get_str(); });
		cdebug << "got secret" << toHex(s);
		BOOST_REQUIRE_EQUAL(toHex(s), o["priv"].get_str());
	}
}

BOOST_AUTO_TEST_SUITE_END()
