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
#include "WhisperHost.h"

using namespace std;
using namespace dev;
using namespace dev::shh;
namespace fs = boost::filesystem;

WhisperDB::WhisperDB(string const& _type)
{
	m_readOptions.verify_checksums = true;
	string path = dev::getDataDir("shh");
	fs::create_directories(path);
	fs::permissions(path, fs::owner_all);
	path += "/" + _type;
	leveldb::Options op;
	op.create_if_missing = true;
	op.max_open_files = 256;
	leveldb::DB* p = nullptr;
	leveldb::Status status = leveldb::DB::Open(op, path, &p);
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

void WhisperMessagesDB::loadAllMessages(std::map<h256, Envelope>& o_dst)
{
	leveldb::ReadOptions op;
	op.fill_cache = false;
	op.verify_checksums = true;
	vector<string> wasted;
	unique_ptr<leveldb::Iterator> it(m_db->NewIterator(op));
	unsigned const now = (unsigned)time(0);

	for (it->SeekToFirst(); it->Valid(); it->Next())
	{
		leveldb::Slice const k = it->key();
		leveldb::Slice const v = it->value();
		bool useless = true;

		try
		{
			RLP rlp((byte const*)v.data(), v.size());
			Envelope e(rlp);
			h256 h2 = e.sha3();
			h256 h1;

			if (k.size() == h256::size)
				h1 = h256((byte const*)k.data(), h256::ConstructFromPointer);

			if (h1 != h2)
				cwarn << "Corrupted data in Level DB:" << h1.hex() << "versus" << h2.hex();
			else if (e.expiry() > now)
			{
				o_dst[h1] = e;
				useless = false;
			}
		}
		catch(RLPException const& ex)
		{
			cwarn << "RLPException in WhisperDB::loadAll():" << ex.what();
		}
		catch(Exception const& ex)
		{
			cwarn << "Exception in WhisperDB::loadAll():" << ex.what();
		}

		if (useless)
			wasted.push_back(k.ToString());
	}

	cdebug << "WhisperDB::loadAll(): loaded " << o_dst.size() << ", deleted " << wasted.size() << "messages";

	for (auto const& k: wasted)
	{
		leveldb::Status status = m_db->Delete(m_writeOptions, k);
		if (!status.ok())
			cwarn << "Failed to delete an entry from Level DB:" << k;
	}
}

void WhisperMessagesDB::saveSingleMessage(h256 const& _key, Envelope const& _e)
{
	try
	{
		RLPStream rlp;
		_e.streamRLP(rlp);
		bytes b;
		rlp.swapOut(b);
		insert(_key, b);
	}
	catch(RLPException const& ex)
	{
		cwarn << boost::diagnostic_information(ex);
	}
	catch(FailedInsertInLevelDB const& ex)
	{
		cwarn << boost::diagnostic_information(ex);
	}
}

vector<unsigned> WhisperFiltersDB::restoreTopicsFromDB(WhisperHost* _host, h256 const& _id)
{
	vector<unsigned> ret;
	string raw = lookup(_id);
	if (!raw.empty())
	{
		RLP rlp(raw);
		auto sz = rlp.itemCountStrict();

		for (unsigned i = 0; i < sz; ++i)
		{
			RLP r = rlp[i];
			bytesConstRef ref(r.toBytesConstRef());
			Topics topics;
			unsigned num = ref.size() / h256::size;
			for (unsigned j = 0; j < num; ++j)
			{
				h256 topic(ref.data() + j * h256::size, h256::ConstructFromPointerType());
				topics.push_back(topic);
			}

			unsigned w = _host->installWatch(topics);
			ret.push_back(w);
		}
	}

	return ret;
}

void WhisperFiltersDB::saveTopicsToDB(WhisperHost const& _host, h256 const& _id)
{
	bytes b;
	RLPStream rlp;
	_host.exportFilters(rlp);
	rlp.swapOut(b);
	insert(_id, b);
}
