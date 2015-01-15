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

/** @file NatspecHandler.cpp
 * @author Lefteris Karapetsas <lefteris@ethdev.com>
 * @date 2015
 */
#include "NatspecHandler.h"
#include <string>
#include <boost/filesystem.hpp>

#include <libdevcore/Common.h>
#include <libdevcore/CommonData.h>
#include <libdevcore/Exceptions.h>
#include <libdevcore/Log.h>
#include <libdevcrypto/SHA3.h>
#include <libethereum/Defaults.h>

using namespace dev;
using namespace dev::eth;
using namespace std;

NatspecHandler::NatspecHandler()
{
	string path = Defaults::dbPath();
	boost::filesystem::create_directories(path);
	ldb::Options o;
	o.create_if_missing = true;
	ldb::DB::Open(o, path + "/natspec", &m_db);
}

NatspecHandler::~NatspecHandler()
{
	delete m_db;
}

void NatspecHandler::add(dev::h256 const& _contractHash, string const& _doc)
{
	bytes k = _contractHash.asBytes();
	string v = _doc;
	m_db->Put(m_writeOptions, ldb::Slice((char const*)k.data(), k.size()), ldb::Slice((char const*)v.data(), v.size()));
}

string NatspecHandler::retrieve(dev::h256 const& _contractHash) const
{
	bytes k = _contractHash.asBytes();
	string ret;
	m_db->Get(m_readOptions, ldb::Slice((char const*)k.data(), k.size()), &ret);
	return ret;
}

string NatspecHandler::getUserNotice(string const& json, dev::bytes const& _transactionData)
{
	Json::Value natspec;
	Json::Value userNotice;
	string retStr;
	m_reader.parse(json, natspec);
	bytes transactionFunctionPart(_transactionData.begin(), _transactionData.begin() + 4);
	FixedHash<4> transactionFunctionHash(transactionFunctionPart);

	Json::Value methods = natspec["methods"];
	for (Json::ValueIterator it = methods.begin(); it != methods.end(); ++it)
	{
		Json::Value keyValue = it.key();
		if (!keyValue.isString())
			BOOST_THROW_EXCEPTION(Exception() << errinfo_comment("Illegal Natspec JSON detected"));

		string functionSig = keyValue.asString();
		FixedHash<4> functionHash(dev::sha3(functionSig));

		if (functionHash == transactionFunctionHash)
		{
			Json::Value val = (*it)["notice"];
			if (!val.isString())
				BOOST_THROW_EXCEPTION(Exception() << errinfo_comment("Illegal Natspec JSON detected"));
			return val.asString();
		}
	}

	// not found
	return string();
}

string NatspecHandler::getUserNotice(dev::h256 const& _contractHash, dev::bytes const& _transactionData)
{
	return getUserNotice(retrieve(_contractHash), _transactionData);
}


