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
/** @file BlockChain.cpp
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 */

#include "BlockChain.h"

#include <boost/filesystem.hpp>
#include <libethsupport/Common.h>
#include <libethsupport/RLP.h>
#include <libethsupport/FileSystem.h>
#include <libethcore/Exceptions.h>
#include <libethcore/Dagger.h>
#include <libethcore/BlockInfo.h>
#include "State.h"
#include "Defaults.h"
using namespace std;
using namespace eth;

#define ETH_CATCH 1

namespace eth
{

std::ostream& operator<<(std::ostream& _out, BlockChain const& _bc)
{
	string cmp = toBigEndianString(_bc.m_lastBlockHash);
	auto it = _bc.m_detailsDB->NewIterator(_bc.m_readOptions);
	for (it->SeekToFirst(); it->Valid(); it->Next())
		if (it->key().ToString() != "best")
		{
			BlockDetails d(RLP(it->value().ToString()));
			_out << toHex(it->key().ToString()) << ":   " << d.number << " @ " << d.parent << (cmp == it->key().ToString() ? "  BEST" : "") << std::endl;
		}
	delete it;
	return _out;
}
}

BlockDetails::BlockDetails(RLP const& _r)
{
	number = _r[0].toInt<uint>();
	totalDifficulty = _r[1].toInt<u256>();
	parent = _r[2].toHash<h256>();
	children = _r[3].toVector<h256>();
}

bytes BlockDetails::rlp() const
{
	return rlpList(number, totalDifficulty, parent, children);
}

std::map<Address, AddressState> const& eth::genesisState()
{
	static std::map<Address, AddressState> s_ret;
	if (s_ret.empty())
		// Initialise.
		for (auto i: vector<string>({
            "51ba59315b3a95761d0863b05ccc7a7f54703d99",
			"e6716f9544a56c530d868e4bfbacb172315bdead",
			"1e12515ce3e0f817a4ddef9ca55788a1d66bd2df",
			"1a26338f0d905e295fccb71fa9ea849ffa12aaf4",
			"2ef47100e0787b915105fd5e3f4ff6752079d5cb",
			"cd2a3d9f938e13cd947ec05abc7fe734df8dd826",
			"6c386a4b26f73c802f34673f7248bb118f97424a",
			"e4157b34ea9615cfbde6b4fda419828124b70c78"
		}))
			s_ret[Address(fromHex(i))] = AddressState(0, u256(1) << 200, h256(), EmptySHA3);
	return s_ret;
}

BlockInfo* BlockChain::s_genesis = nullptr;

bytes BlockChain::createGenesisBlock()
{
	RLPStream block(3);
	auto sha3EmptyList = sha3(RLPEmptyList);

	h256 stateRoot;
	{
		MemoryDB db;
		TrieDB<Address, MemoryDB> state(&db);
		state.init();
		eth::commit(genesisState(), db, state);
		stateRoot = state.root();
	}

	block.appendList(13) << h256() << sha3EmptyList << h160();
	block.append(stateRoot, false, true) << bytes() << c_genesisDifficulty << 0 << 0 << 1000000 << 0 << (uint)0 << string() << sha3(bytes(1, 42));
	block.appendRaw(RLPEmptyList);
	block.appendRaw(RLPEmptyList);
	return block.out();
}

BlockChain::BlockChain(std::string _path, bool _killExisting)
{
	if (_path.empty())
		_path = Defaults::get()->m_dbPath;
	boost::filesystem::create_directories(_path);
	if (_killExisting)
	{
		boost::filesystem::remove_all(_path + "/blocks");
		boost::filesystem::remove_all(_path + "/details");
	}

	ldb::Options o;
	o.create_if_missing = true;
	auto s = ldb::DB::Open(o, _path + "/blocks", &m_db);
	assert(m_db);
	s = ldb::DB::Open(o, _path + "/details", &m_detailsDB);
	assert(m_detailsDB);

	// Initialise with the genesis as the last block on the longest chain.
	m_genesisHash = BlockChain::genesis().hash;
	m_genesisBlock = BlockChain::createGenesisBlock();

	if (!details(m_genesisHash))
	{
		// Insert details of genesis block.
		m_details[m_genesisHash] = BlockDetails(0, c_genesisDifficulty, h256(), {});
		auto r = m_details[m_genesisHash].rlp();
		m_detailsDB->Put(m_writeOptions, ldb::Slice((char const*)&m_genesisHash, 32), (ldb::Slice)eth::ref(r));
	}

	checkConsistency();

	// TODO: Implement ability to rebuild details map from DB.
	std::string l;
	m_detailsDB->Get(m_readOptions, ldb::Slice("best"), &l);
	m_lastBlockHash = l.empty() ? m_genesisHash : *(h256*)l.data();

	cnote << "Opened blockchain DB. Latest: " << m_lastBlockHash;
}

BlockChain::~BlockChain()
{
	cnote << "Closing blockchain DB";
	delete m_detailsDB;
	delete m_db;
}

template <class T, class V>
bool contains(T const& _t, V const& _v)
{
	for (auto const& i: _t)
		if (i == _v)
			return true;
	return false;
}

bool BlockChain::attemptImport(bytes const& _block, OverlayDB const& _stateDB)
{
#if ETH_CATCH
	try
#endif
	{
		import(_block, _stateDB);
		return true;
	}
#if ETH_CATCH
	catch (...)
	{
		return false;
	}
#endif
}


void BlockChain::import(bytes const& _block, OverlayDB const& _db)
{
	// VERIFY: populates from the block and checks the block is internally coherent.
	BlockInfo bi;

#if ETH_CATCH
	try
#endif
	{
		bi.populate(&_block);
		bi.verifyInternals(&_block);
	}
#if ETH_CATCH
	catch (Exception const& _e)
	{
		clog(BlockChainNote) << "   Malformed block (" << _e.description() << ").";
		throw;
	}
#endif
	auto newHash = eth::sha3(_block);

	// Check block doesn't already exist first!
	if (details(newHash))
	{
		clog(BlockChainNote) << newHash << ": Not new.";
		throw AlreadyHaveBlock();
	}

	// Work out its number as the parent's number + 1
	auto pd = details(bi.parentHash);
	if (!pd)
	{
		clog(BlockChainNote) << newHash << ": Unknown parent " << bi.parentHash;
		// We don't know the parent (yet) - discard for now. It'll get resent to us if we find out about its ancestry later on.
		throw UnknownParent();
	}

	// Check it's not crazy
	if (bi.timestamp > (u256)time(0))
	{
		clog(BlockChainNote) << newHash << ": Future time " << bi.timestamp << " (now at " << time(0) << ")";
		// We don't know the parent (yet) - discard for now. It'll get resent to us if we find out about its ancestry later on.
		throw FutureTime();
	}

	clog(BlockChainNote) << "Attempting import of " << newHash << "...";

	u256 td;
#if ETH_CATCH
	try
#endif
	{
		// Check family:
		BlockInfo biParent(block(bi.parentHash));
		bi.verifyParent(biParent);

		// Check transactions are valid and that they result in a state equivalent to our state_root.
		State s(bi.coinbaseAddress, _db);
		s.sync(*this, bi.parentHash);

		// Get total difficulty increase and update state, checking it.
		BlockInfo biGrandParent;
		if (pd.number)
			biGrandParent.populate(block(pd.parent));
		auto tdIncrease = s.playback(&_block, bi, biParent, biGrandParent, true);
		td = pd.totalDifficulty + tdIncrease;

#if ETH_PARANOIA
		checkConsistency();
#endif
		// All ok - insert into DB
		{
			lock_guard<mutex> l(m_lock);
			m_details[newHash] = BlockDetails((uint)pd.number + 1, td, bi.parentHash, {});
			m_details[bi.parentHash].children.push_back(newHash);
		}

		m_detailsDB->Put(m_writeOptions, ldb::Slice((char const*)&newHash, 32), (ldb::Slice)eth::ref(m_details[newHash].rlp()));
		m_detailsDB->Put(m_writeOptions, ldb::Slice((char const*)&bi.parentHash, 32), (ldb::Slice)eth::ref(m_details[bi.parentHash].rlp()));
		m_db->Put(m_writeOptions, ldb::Slice((char const*)&newHash, 32), (ldb::Slice)ref(_block));

#if ETH_PARANOIA
		checkConsistency();
#endif
	}
#if ETH_CATCH
	catch (Exception const& _e)
	{
		clog(BlockChainNote) << "   Malformed block (" << _e.description() << ").";
		throw;
	}
#endif

//	cnote << "Parent " << bi.parentHash << " has " << details(bi.parentHash).children.size() << " children.";

	// This might be the new best block...
	if (td > details(m_lastBlockHash).totalDifficulty)
	{
		m_lastBlockHash = newHash;
		m_detailsDB->Put(m_writeOptions, ldb::Slice("best"), ldb::Slice((char const*)&newHash, 32));
		clog(BlockChainNote) << "   Imported and best. Has" << (details(bi.parentHash).children.size() - 1) << "siblings.";
	}
	else
	{
		clog(BlockChainNote) << "   Imported but not best (oTD:" << details(m_lastBlockHash).totalDifficulty << ", TD:" << td << ")";
	}
}

void BlockChain::checkConsistency()
{
	m_details.clear();
	ldb::Iterator* it = m_detailsDB->NewIterator(m_readOptions);
	for (it->SeekToFirst(); it->Valid(); it->Next())
		if (it->key().size() == 32)
		{
			h256 h((byte const*)it->key().data(), h256::ConstructFromPointer);
			auto dh = details(h);
			auto p = dh.parent;
			if (p != h256())
			{
				auto dp = details(p);
				assert(contains(dp.children, h));
				assert(dp.number == dh.number - 1);
			}
		}
	delete it;
}

bytesConstRef BlockChain::block(h256 _hash) const
{
	if (_hash == m_genesisHash)
		return &m_genesisBlock;

	string d;
	m_db->Get(m_readOptions, ldb::Slice((char const*)&_hash, 32), &d);

	{
		lock_guard<mutex> l(m_lock);
		swap(m_cache[_hash], d);
		return bytesConstRef(&m_cache[_hash]);
	}
}

eth::uint BlockChain::number(h256 _hash) const
{
	return details(_hash).number;
}

BlockDetails const& BlockChain::details(h256 _h) const
{
	BlockDetailsHash::const_iterator it;
	bool fetchRequired;
	{
		lock_guard<mutex> l(m_lock);
		it = m_details.find(_h);
		fetchRequired = (it == m_details.end());
	}
	if (fetchRequired)
	{
		std::string s;
		m_detailsDB->Get(m_readOptions, ldb::Slice((char const*)&_h, 32), &s);
		if (s.empty())
		{
//			cout << "Not found in DB: " << _h << endl;
			return NullBlockDetails;
		}
		{
			lock_guard<mutex> l(m_lock);
			bool ok;
			tie(it, ok) = m_details.insert(std::make_pair(_h, BlockDetails(RLP(s))));
		}
	}
	return it->second;
}
