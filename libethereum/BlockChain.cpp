/*
	This file is part of cpp-ethereum.

	cpp-ethereum is free software: you can redistribute it and/or modify
	it under the terms of the GNU General Public License as published by
	the Free Software Foundation, either version 3 of the License, or
	(at your option) any later version.

	Foobar is distributed in the hope that it will be useful,
	but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	GNU General Public License for more details.

	You should have received a copy of the GNU General Public License
	along with Foobar.  If not, see <http://www.gnu.org/licenses/>.
*/
/** @file BlockChain.cpp
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 */

#include <boost/filesystem.hpp>
#include "Common.h"
#include "RLP.h"
#include "Exceptions.h"
#include "Dagger.h"
#include "BlockInfo.h"
#include "State.h"
#include "BlockChain.h"
using namespace std;
using namespace eth;

std::string Defaults::s_dbPath = string(getenv("HOME")) + "/.ethereum";

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
			_out << asHex(it->key().ToString()) << ":   " << d.number << " @ " << d.parent << (cmp == it->key().ToString() ? "  BEST" : "") << std::endl;
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

BlockChain::BlockChain(std::string _path, bool _killExisting)
{
	if (_path.empty())
		_path = Defaults::s_dbPath;
	boost::filesystem::create_directory(_path);
	if (_killExisting)
	{
		boost::filesystem::remove_all(_path + "/blocks");
		boost::filesystem::remove_all(_path + "/details");
	}

	ldb::Options o;
	o.create_if_missing = true;
	auto s = ldb::DB::Open(o, _path + "/blocks", &m_db);
	s = ldb::DB::Open(o, _path + "/details", &m_detailsDB);

	// Initialise with the genesis as the last block on the longest chain.
	m_genesisHash = BlockInfo::genesis().hash;
	m_genesisBlock = BlockInfo::createGenesisBlock();

	// Insert details of genesis block.
	m_details[m_genesisHash] = BlockDetails(0, (u256)1 << 36, h256(), {});

	// TODO: Implement ability to rebuild details map from DB.
	std::string l;
	m_detailsDB->Get(m_readOptions, ldb::Slice("best"), &l);
	m_lastBlockHash = l.empty() ? m_genesisHash : *(h256*)l.data();
}

BlockChain::~BlockChain()
{
}

void BlockChain::import(bytes const& _block, Overlay const& _db)
{
	// VERIFY: populates from the block and checks the block is internally coherent.
	BlockInfo bi(&_block);
	bi.verifyInternals(&_block);

	auto newHash = eth::sha3(_block);

	// Check block doesn't already exist first!
	if (details(newHash))
		throw AlreadyHaveBlock();

	// Work out its number as the parent's number + 1
	auto pd = details(bi.parentHash);
	if (!pd)
		// We don't know the parent (yet) - discard for now. It'll get resent to us if we find out about its ancestry later on.
		throw UnknownParent();

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
	u256 td = pd.totalDifficulty + tdIncrease;

	// All ok - insert into DB
	m_details[newHash] = BlockDetails((uint)pd.number + 1, td, bi.parentHash, {});
	m_detailsDB->Put(m_writeOptions, ldb::Slice((char const*)&newHash, 32), (ldb::Slice)eth::ref(m_details[newHash].rlp()));

	m_details[bi.parentHash].children.push_back(newHash);
	m_detailsDB->Put(m_writeOptions, ldb::Slice((char const*)&bi.parentHash, 32), (ldb::Slice)eth::ref(m_details[bi.parentHash].rlp()));

	m_db->Put(m_writeOptions, ldb::Slice((char const*)&newHash, 32), (ldb::Slice)ref(_block));

	// This might be the new last block...
	if (td > m_details[m_lastBlockHash].totalDifficulty)
	{
		m_lastBlockHash = newHash;
		m_detailsDB->Put(m_writeOptions, ldb::Slice("best"), ldb::Slice((char const*)&newHash, 32));
	}
	else
	{
		cerr << "*** WARNING: Imported block not newest (otd=" << m_details[m_lastBlockHash].totalDifficulty << ", td=" << td << ")" << endl;
	}
}

bytesConstRef BlockChain::block(h256 _hash) const
{
	if (_hash == m_genesisHash)
		return &m_genesisBlock;

	m_db->Get(m_readOptions, ldb::Slice((char const*)&_hash, 32), &m_cache[_hash]);
	return bytesConstRef(&m_cache[_hash]);
}

BlockDetails const& BlockChain::details(h256 _h) const
{
	auto it = m_details.find(_h);
	if (it == m_details.end())
	{
		std::string s;
		m_detailsDB->Get(m_readOptions, ldb::Slice((char const*)&_h, 32), &s);
		if (s.empty())
			return NullBlockDetails;
		bool ok;
		tie(it, ok) = m_details.insert(make_pair(_h, BlockDetails(RLP(s))));
	}
	return it->second;
}
