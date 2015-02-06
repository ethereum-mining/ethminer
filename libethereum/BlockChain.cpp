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
#include <libdevcore/Common.h>
#include <libdevcore/RLP.h>
#include <libdevcrypto/FileSystem.h>
#include <libethcore/Exceptions.h>
#include <libethcore/ProofOfWork.h>
#include <libethcore/BlockInfo.h>
#include <liblll/Compiler.h>
#include "State.h"
#include "Defaults.h"
using namespace std;
using namespace dev;
using namespace dev::eth;

#define ETH_CATCH 1

std::ostream& dev::eth::operator<<(std::ostream& _out, BlockChain const& _bc)
{
	string cmp = toBigEndianString(_bc.currentHash());
	auto it = _bc.m_extrasDB->NewIterator(_bc.m_readOptions);
	for (it->SeekToFirst(); it->Valid(); it->Next())
		if (it->key().ToString() != "best")
		{
			string rlpString = it->value().ToString();
			RLP r(rlpString);
			BlockDetails d(r);
			_out << toHex(it->key().ToString()) << ":   " << d.number << " @ " << d.parent << (cmp == it->key().ToString() ? "  BEST" : "") << std::endl;
		}
	delete it;
	return _out;
}

ldb::Slice dev::eth::toSlice(h256 _h, unsigned _sub)
{
#if ALL_COMPILERS_ARE_CPP11_COMPLIANT
	static thread_local h256 h = _h ^ h256(u256(_sub));
	return ldb::Slice((char const*)&h, 32);
#else
	static boost::thread_specific_ptr<h256> t_h;
	if (!t_h.get())
		t_h.reset(new h256);
	*t_h = _h ^ h256(u256(_sub));
	return ldb::Slice((char const*)t_h.get(), 32);
#endif
}

BlockChain::BlockChain(bytes const& _genesisBlock, std::string _path, bool _killExisting)
{
	// Initialise with the genesis as the last block on the longest chain.
	m_genesisBlock = _genesisBlock;
	m_genesisHash = sha3(RLP(m_genesisBlock)[0].data());

	open(_path, _killExisting);
}

BlockChain::~BlockChain()
{
	close();
}

void BlockChain::open(std::string _path, bool _killExisting)
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
	ldb::DB::Open(o, _path + "/blocks", &m_db);
	ldb::DB::Open(o, _path + "/details", &m_extrasDB);
	if (!m_db)
		BOOST_THROW_EXCEPTION(DatabaseAlreadyOpen());
	if (!m_extrasDB)
		BOOST_THROW_EXCEPTION(DatabaseAlreadyOpen());

	if (!details(m_genesisHash))
	{
		// Insert details of genesis block.
		m_details[m_genesisHash] = BlockDetails(0, c_genesisDifficulty, h256(), {});
		auto r = m_details[m_genesisHash].rlp();
		m_extrasDB->Put(m_writeOptions, ldb::Slice((char const*)&m_genesisHash, 32), (ldb::Slice)dev::ref(r));
	}

	checkConsistency();

	// TODO: Implement ability to rebuild details map from DB.
	std::string l;
	m_extrasDB->Get(m_readOptions, ldb::Slice("best"), &l);

	m_lastBlockHash = l.empty() ? m_genesisHash : *(h256*)l.data();

	cnote << "Opened blockchain DB. Latest: " << currentHash();
}

void BlockChain::close()
{
	cnote << "Closing blockchain DB";
	delete m_extrasDB;
	delete m_db;
	m_lastBlockHash = m_genesisHash;
	m_details.clear();
	m_cache.clear();
}

template <class T, class V>
bool contains(T const& _t, V const& _v)
{
	for (auto const& i: _t)
		if (i == _v)
			return true;
	return false;
}

inline string toString(h256s const& _bs)
{
	ostringstream out;
	out << "[ ";
	for (auto i: _bs)
		out << i.abridged() << ", ";
	out << "]";
	return out.str();
}

h256s BlockChain::sync(BlockQueue& _bq, OverlayDB const& _stateDB, unsigned _max)
{
	_bq.tick(*this);

	vector<bytes> blocks;
	_bq.drain(blocks);

	h256s ret;
	for (auto const& block: blocks)
	{
		try
		{
			for (auto h: import(block, _stateDB))
				if (!_max--)
					break;
				else
					ret.push_back(h);
		}
		catch (UnknownParent)
		{
			cwarn << "Unknown parent of block!!!" << BlockInfo::headerHash(block).abridged() << boost::current_exception_diagnostic_information();
			_bq.import(&block, *this);
		}
		catch (Exception const& _e)
		{
			cwarn << "Unexpected exception!" << diagnostic_information(_e);
			_bq.import(&block, *this);
		}
		catch (...)
		{}
	}
	_bq.doneDrain();
	return ret;
}

h256s BlockChain::attemptImport(bytes const& _block, OverlayDB const& _stateDB) noexcept
{
	try
	{
		return import(_block, _stateDB);
	}
	catch (...)
	{
		cwarn << "Unexpected exception! Could not import block!" << boost::current_exception_diagnostic_information();
		return h256s();
	}
}

h256s BlockChain::import(bytes const& _block, OverlayDB const& _db)
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
		clog(BlockChainNote) << "   Malformed block: " << diagnostic_information(_e);
		_e << errinfo_comment("Malformed block ");
		throw;
	}
#endif
	auto newHash = BlockInfo::headerHash(_block);

	// Check block doesn't already exist first!
	if (isKnown(newHash))
	{
		clog(BlockChainNote) << newHash << ": Not new.";
		BOOST_THROW_EXCEPTION(AlreadyHaveBlock());
	}

	// Work out its number as the parent's number + 1
	if (!isKnown(bi.parentHash))
	{
		clog(BlockChainNote) << newHash << ": Unknown parent " << bi.parentHash;
		// We don't know the parent (yet) - discard for now. It'll get resent to us if we find out about its ancestry later on.
		BOOST_THROW_EXCEPTION(UnknownParent());
	}

	auto pd = details(bi.parentHash);
	if (!pd)
	{
		auto pdata = pd.rlp();
		cwarn << "Odd: details is returning false despite block known:" << RLP(pdata);
		auto parentBlock = block(bi.parentHash);
		cwarn << "Block:" << RLP(parentBlock);
	}

	// Check it's not crazy
	if (bi.timestamp > (u256)time(0))
	{
		clog(BlockChainNote) << newHash << ": Future time " << bi.timestamp << " (now at " << time(0) << ")";
		// Block has a timestamp in the future. This is no good.
		BOOST_THROW_EXCEPTION(FutureTime());
	}

	clog(BlockChainNote) << "Attempting import of " << newHash.abridged() << "...";

	u256 td;
#if ETH_CATCH
	try
#endif
	{
		// Check transactions are valid and that they result in a state equivalent to our state_root.
		// Get total difficulty increase and update state, checking it.
		State s(bi.coinbaseAddress, _db);
		auto tdIncrease = s.enactOn(&_block, bi, *this);
		BlockLogBlooms blb;
		BlockReceipts br;
		for (unsigned i = 0; i < s.pending().size(); ++i)
		{
			blb.blooms.push_back(s.receipt(i).bloom());
			br.receipts.push_back(s.receipt(i));
		}
		s.cleanup(true);
		td = pd.totalDifficulty + tdIncrease;

#if ETH_PARANOIA
		checkConsistency();
#endif
		// All ok - insert into DB
		{
			WriteGuard l(x_details);
			m_details[newHash] = BlockDetails((unsigned)pd.number + 1, td, bi.parentHash, {});
			m_details[bi.parentHash].children.push_back(newHash);
		}
		{
			WriteGuard l(x_logBlooms);
			m_logBlooms[newHash] = blb;
		}
		{
			WriteGuard l(x_receipts);
			m_receipts[newHash] = br;
		}

		m_extrasDB->Put(m_writeOptions, toSlice(newHash), (ldb::Slice)dev::ref(m_details[newHash].rlp()));
		m_extrasDB->Put(m_writeOptions, toSlice(bi.parentHash), (ldb::Slice)dev::ref(m_details[bi.parentHash].rlp()));
		m_extrasDB->Put(m_writeOptions, toSlice(newHash, 3), (ldb::Slice)dev::ref(m_logBlooms[newHash].rlp()));
		m_extrasDB->Put(m_writeOptions, toSlice(newHash, 4), (ldb::Slice)dev::ref(m_receipts[newHash].rlp()));
		m_db->Put(m_writeOptions, toSlice(newHash), (ldb::Slice)ref(_block));

#if ETH_PARANOIA
		checkConsistency();
#endif
	}
#if ETH_CATCH
	catch (Exception const& _e)
	{
		clog(BlockChainNote) << "   Malformed block: " << diagnostic_information(_e);
		_e << errinfo_comment("Malformed block ");
		throw;
	}
#endif

	//	cnote << "Parent " << bi.parentHash << " has " << details(bi.parentHash).children.size() << " children.";

	h256s ret;
	// This might be the new best block...
	h256 last = currentHash();
	if (td > details(last).totalDifficulty)
	{
		ret = treeRoute(last, newHash);
		{
			WriteGuard l(x_lastBlockHash);
			m_lastBlockHash = newHash;
		}
		m_extrasDB->Put(m_writeOptions, ldb::Slice("best"), ldb::Slice((char const*)&newHash, 32));
		clog(BlockChainNote) << "   Imported and best" << td << ". Has" << (details(bi.parentHash).children.size() - 1) << "siblings. Route:" << toString(ret);
	}
	else
	{
		clog(BlockChainNote) << "   Imported but not best (oTD:" << details(last).totalDifficulty << " > TD:" << td << ")";
	}
	return ret;
}

h256s BlockChain::treeRoute(h256 _from, h256 _to, h256* o_common, bool _pre, bool _post) const
{
	//	cdebug << "treeRoute" << _from.abridged() << "..." << _to.abridged();
	if (!_from || !_to)
	{
		return h256s();
	}
	h256s ret;
	h256s back;
	unsigned fn = details(_from).number;
	unsigned tn = details(_to).number;
	//	cdebug << "treeRoute" << fn << "..." << tn;
	while (fn > tn)
	{
		if (_pre)
			ret.push_back(_from);
		_from = details(_from).parent;
		fn--;
		//		cdebug << "from:" << fn << _from.abridged();
	}
	while (fn < tn)
	{
		if (_post)
			back.push_back(_to);
		_to = details(_to).parent;
		tn--;
		//		cdebug << "to:" << tn << _to.abridged();
	}
	while (_from != _to)
	{
		assert(_from);
		assert(_to);
		_from = details(_from).parent;
		_to = details(_to).parent;
		if (_pre)
			ret.push_back(_from);
		if (_post)
			back.push_back(_to);
		fn--;
		tn--;
		//		cdebug << "from:" << fn << _from.abridged() << "; to:" << tn << _to.abridged();
	}
	if (o_common)
		*o_common = _from;
	ret.reserve(ret.size() + back.size());
	for (auto it = back.cbegin(); it != back.cend(); ++it)
		ret.push_back(*it);
	return ret;
}

void BlockChain::checkConsistency()
{
	{
		WriteGuard l(x_details);
		m_details.clear();
	}
	ldb::Iterator* it = m_db->NewIterator(m_readOptions);
	for (it->SeekToFirst(); it->Valid(); it->Next())
		if (it->key().size() == 32)
		{
			h256 h((byte const*)it->key().data(), h256::ConstructFromPointer);
			auto dh = details(h);
			auto p = dh.parent;
			if (p != h256() && p != m_genesisHash)	// TODO: for some reason the genesis details with the children get squished. not sure why.
			{
				auto dp = details(p);
				if (asserts(contains(dp.children, h)))
				{
					cnote << "Apparently the database is corrupt. Not much we can do at this stage...";
				}
				if (assertsEqual(dp.number, dh.number - 1))
				{
					cnote << "Apparently the database is corrupt. Not much we can do at this stage...";
				}
			}
		}
	delete it;
}

h256Set BlockChain::allUnclesFrom(h256 _parent) const
{
	// Get all uncles cited given a parent (i.e. featured as uncles/main in parent, parent + 1, ... parent + 5).
	h256Set ret;
	h256 p = _parent;
	for (unsigned i = 0; i < 6 && p != m_genesisHash; ++i, p = details(p).parent)
	{
		ret.insert(p);		// TODO: check: should this be details(p).parent?
		auto b = block(p);
		for (auto i: RLP(b)[2])
			ret.insert(sha3(i.data()));
	}
	return ret;
}

bool BlockChain::isKnown(h256 _hash) const
{
	if (_hash == m_genesisHash)
		return true;
	{
		ReadGuard l(x_cache);
		if (m_cache.count(_hash))
			return true;
	}
	string d;
	m_db->Get(m_readOptions, ldb::Slice((char const*)&_hash, 32), &d);
	return !!d.size();
}

bytes BlockChain::block(h256 _hash) const
{
	if (_hash == m_genesisHash)
		return m_genesisBlock;

	{
		ReadGuard l(x_cache);
		auto it = m_cache.find(_hash);
		if (it != m_cache.end())
			return it->second;
	}

	string d;
	m_db->Get(m_readOptions, ldb::Slice((char const*)&_hash, 32), &d);

	if (!d.size())
	{
		cwarn << "Couldn't find requested block:" << _hash.abridged();
		return bytes();
	}

	WriteGuard l(x_cache);
	m_cache[_hash].resize(d.size());
	memcpy(m_cache[_hash].data(), d.data(), d.size());

	return m_cache[_hash];
}

h256 BlockChain::numberHash(unsigned _n) const
{
	if (!_n)
		return genesisHash();
	h256 ret = currentHash();
	for (; _n < details().number; ++_n, ret = details(ret).parent) {}
	return ret;
}
