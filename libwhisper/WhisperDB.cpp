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

#include "WhisperDB.h"
#include <boost/filesystem.hpp>
#include <libdevcore/FileSystem.h>

using namespace std;
using namespace dev;
using namespace dev::shh;

WhisperDB::WhisperDB()
{
	string path = dev::getDataDir();
	boost::filesystem::create_directories(path);
	leveldb::Options op;
	op.create_if_missing = true;
	op.max_open_files = 256;
	leveldb::DB* p = nullptr;
	leveldb::Status status = leveldb::DB::Open(op, path + "/whisper", &p);
	m_db.reset(p);
	if (!status.ok())
		BOOST_THROW_EXCEPTION(FailedToOpenLevelDB(status.ToString()));
}

string WhisperDB::lookup(dev::h256 const& _key) const
{
	string ret;
	leveldb::Slice slice((char const*)_key.data(), _key.size);
	leveldb::Status status = m_db->Get(m_readOptions, slice, &ret);
	if (!status.ok() && !status.IsNotFound())
		BOOST_THROW_EXCEPTION(FailedLookupInLevelDB(status.ToString()));

	return ret;
}

void WhisperDB::insert(dev::h256 const& _key, string const& _value)
{
	leveldb::Slice slice((char const*)_key.data(), _key.size);
	leveldb::Status status = m_db->Put(m_writeOptions, slice, _value);	
	if (!status.ok())
		BOOST_THROW_EXCEPTION(FailedInsertInLevelDB(status.ToString()));
}

void WhisperDB::kill(dev::h256 const& _key)
{
	leveldb::Slice slice((char const*)_key.data(), _key.size);
	leveldb::Status status = m_db->Delete(m_writeOptions, slice);
	if (!status.ok())
		BOOST_THROW_EXCEPTION(FailedDeleteInLevelDB(status.ToString()));
}
