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
	string path = dev::getDataDir("shh");
	boost::filesystem::create_directories(path);
	leveldb::Options op;
	op.create_if_missing = true;
	op.max_open_files = 256;
	leveldb::DB* p = nullptr;
	leveldb::Status status = leveldb::DB::Open(op, path + "/messages", &p);
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

void WhisperDB::insert(dev::h256 const& _key, bytes const& _value)
{
	leveldb::Slice k((char const*)_key.data(), _key.size);
	leveldb::Slice v((char const*)_value.data(), _value.size());
	leveldb::Status status = m_db->Put(m_writeOptions, k, v);
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

void WhisperDB::loadAll(std::map<h256, Envelope>& o_dst)
{
	leveldb::ReadOptions op;
	op.fill_cache = false;
	op.verify_checksums = true;
	vector<leveldb::Slice> wasted;
	unsigned now = (unsigned)time(0);
	leveldb::Iterator* it = m_db->NewIterator(op);

	for (it->SeekToFirst(); it->Valid(); it->Next())
	{
		leveldb::Slice const k = it->key();
		leveldb::Slice const v = it->value();

		bool useless = false;
		RLP rlp((byte const*)v.data(), v.size());
		Envelope e(rlp);
		h256 h2 = e.sha3();
		h256 h1;

		if (k.size() == h256::size)
			h1 = h256((byte const*)k.data(), h256::ConstructFromPointer);

		if (h1 != h2)
		{
			useless = true;
			cwarn << "Corrupted data in Level DB:" << h1.hex() << "versus" << h2.hex();
		}
		else if (e.expiry() <= now)
			useless = true;

		if (useless)
			wasted.push_back(k);
		else
			o_dst[h1] = e;
	}

	leveldb::WriteOptions woptions;
	for (auto k: wasted)
		m_db->Delete(woptions, k);
}
