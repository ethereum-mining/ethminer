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
/** @file WhisperDB.cpp
* @author Vladislav Gluhovsky <vlad@ethdev.com>
* @date July 2015
 */

#include <string>
#include <boost/filesystem.hpp>

#include <libdevcore/Common.h>
#include <libdevcore/CommonData.h>
#include <libdevcore/Exceptions.h>
#include <libdevcore/Log.h>
#include <libdevcore/SHA3.h>
#include <libethereum/Defaults.h>
#include "WhisperDB.h"

using namespace std;
using namespace dev;
using namespace dev::shh;
using namespace dev::eth;

WhisperDB::WhisperDB()
{
	string path = Defaults::dbPath();
	boost::filesystem::create_directories(path);
	ldb::Options o;
	o.create_if_missing = true;
	ldb::DB::Open(o, path + "/whisper", &m_db);
}

WhisperDB::~WhisperDB()
{
	delete m_db;
}

bool WhisperDB::put(dev::h256 const& _key, string const& _value)
{
	string s = _key.hex();
	string cropped = s.substr(s.size() - 8);
	leveldb::Status status = m_db->Put(m_writeOptions, s, _value);
	if (status.ok())
		cdebug << "Whisper DB put:" << cropped << _value;
	else
		cdebug << "Whisper DB put failed:" << status.ToString() << "key:" << cropped;

	return status.ok();
}

string WhisperDB::get(dev::h256 const& _key) const
{
	string ret;
	string s = _key.hex();
	string cropped = s.substr(s.size() - 8);
	leveldb::Status status = m_db->Get(m_readOptions, s, &ret);
	if (status.ok())
		cdebug << "Whisper DB get:" << cropped << ret;
	else
		cdebug << "Whisper DB get failed:" << status.ToString() << "key:" << cropped;

	return ret;
}

bool WhisperDB::erase(dev::h256 const& _key)
{
	string s = _key.hex();
	string cropped = s.substr(s.size() - 8);
	leveldb::Status status = m_db->Delete(m_writeOptions, s);
	if (status.ok())
		cdebug << "Whisper DB erase:" << cropped;
	else
		cdebug << "Whisper DB erase failed:" << status.ToString() << "key:" << cropped;
	
	return status.ok();
}
