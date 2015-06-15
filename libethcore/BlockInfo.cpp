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
/** @file BlockInfo.cpp
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 */

#include <libdevcore/Common.h>
#include <libdevcore/RLP.h>
#include <libdevcore/TrieDB.h>
#include <libdevcore/TrieHash.h>
#include <libethcore/Common.h>
#include <libethcore/Params.h>
#include "EthashAux.h"
#include "ProofOfWork.h"
#include "Exceptions.h"
#include "BlockInfo.h"
using namespace std;
using namespace dev;
using namespace dev::eth;

BlockInfo::BlockInfo(): timestamp(Invalid256)
{
}

BlockInfo::BlockInfo(bytesConstRef _block, Strictness _s, h256 const& _h)
{
	populate(_block, _s, _h);
}

void BlockInfo::clear()
{
	parentHash = h256();
	sha3Uncles = EmptyListSHA3;
	coinbaseAddress = Address();
	stateRoot = EmptyTrie;
	transactionsRoot = EmptyTrie;
	receiptsRoot = EmptyTrie;
	logBloom = LogBloom();
	difficulty = 0;
	number = 0;
	gasLimit = 0;
	gasUsed = 0;
	timestamp = 0;
	extraData.clear();
	mixHash = h256();
	nonce = Nonce();
	m_hash = m_seedHash = h256();
}

h256 const& BlockInfo::seedHash() const
{
	if (!m_seedHash)
		m_seedHash = EthashAux::seedHash((unsigned)number);
	return m_seedHash;
}

h256 const& BlockInfo::hash() const
{
	if (!m_hash)
		m_hash = headerHash(WithNonce);
	return m_hash;
}

h256 const& BlockInfo::boundary() const
{
	if (!m_boundary && difficulty)
		m_boundary = (h256)(u256)((bigint(1) << 256) / difficulty);
	return m_boundary;
}

BlockInfo BlockInfo::fromHeader(bytesConstRef _header, Strictness _s, h256 const& _h)
{
	BlockInfo ret;
	ret.populateFromHeader(RLP(_header), _s, _h);
	return ret;
}

h256 BlockInfo::headerHash(IncludeNonce _n) const
{
	RLPStream s;
	streamRLP(s, _n);
	return sha3(s.out());
}

void BlockInfo::streamRLP(RLPStream& _s, IncludeNonce _n) const
{
	_s.appendList(_n == WithNonce ? 15 : 13)
		<< parentHash << sha3Uncles << coinbaseAddress << stateRoot << transactionsRoot << receiptsRoot << logBloom
		<< difficulty << number << gasLimit << gasUsed << timestamp << extraData;
	if (_n == WithNonce)
		_s << mixHash << nonce;
}

h256 BlockInfo::headerHash(bytesConstRef _block)
{
	return sha3(RLP(_block)[0].data());
}

void BlockInfo::populateFromHeader(RLP const& _header, Strictness _s, h256 const& _h)
{
	m_hash = _h;
	if (_h)
		assert(_h == dev::sha3(_header.data()));
	m_seedHash = h256();

	int field = 0;
	try
	{
		if (_header.itemCount() != 15)
			throw InvalidBlockHeaderItemCount();
		parentHash = _header[field = 0].toHash<h256>(RLP::VeryStrict);
		sha3Uncles = _header[field = 1].toHash<h256>(RLP::VeryStrict);
		coinbaseAddress = _header[field = 2].toHash<Address>(RLP::VeryStrict);
		stateRoot = _header[field = 3].toHash<h256>(RLP::VeryStrict);
		transactionsRoot = _header[field = 4].toHash<h256>(RLP::VeryStrict);
		receiptsRoot = _header[field = 5].toHash<h256>(RLP::VeryStrict);
		logBloom = _header[field = 6].toHash<LogBloom>(RLP::VeryStrict);
		difficulty = _header[field = 7].toInt<u256>();
		number = _header[field = 8].toInt<u256>();
		gasLimit = _header[field = 9].toInt<u256>();
		gasUsed = _header[field = 10].toInt<u256>();
		timestamp = _header[field = 11].toInt<u256>();
		extraData = _header[field = 12].toBytes();
		mixHash = _header[field = 13].toHash<h256>(RLP::VeryStrict);
		nonce = _header[field = 14].toHash<Nonce>(RLP::VeryStrict);
	}
	catch (Exception const& _e)
	{
		_e << errinfo_name("invalid block header format") << BadFieldError(field, toHex(_header[field].data().toBytes()));
		throw;
	}

	if (number > ~(unsigned)0)
		throw InvalidNumber();

	// check it hashes according to proof of work or that it's the genesis block.
	if (_s == CheckEverything && parentHash && !ProofOfWork::verify(*this))
	{
		InvalidBlockNonce ex;
		ex << errinfo_hash256(headerHash(WithoutNonce));
		ex << errinfo_nonce(nonce);
		ex << errinfo_difficulty(difficulty);
		ex << errinfo_seedHash(seedHash());
		ex << errinfo_target(boundary());
		ex << errinfo_mixHash(mixHash);
		Ethash::Result er = EthashAux::eval(seedHash(), headerHash(WithoutNonce), nonce);
		ex << errinfo_ethashResult(make_tuple(er.value, er.mixHash));
		BOOST_THROW_EXCEPTION(ex);
	}
	else if (_s == QuickNonce && parentHash && !ProofOfWork::preVerify(*this))
	{
		InvalidBlockNonce ex;
		ex << errinfo_hash256(headerHash(WithoutNonce));
		ex << errinfo_nonce(nonce);
		ex << errinfo_difficulty(difficulty);
		BOOST_THROW_EXCEPTION(ex);
	}

	if (_s != CheckNothing)
	{
		if (gasUsed > gasLimit)
			BOOST_THROW_EXCEPTION(TooMuchGasUsed() << RequirementError(bigint(gasLimit), bigint(gasUsed)) );

		if (difficulty < c_minimumDifficulty)
			BOOST_THROW_EXCEPTION(InvalidDifficulty() << RequirementError(bigint(c_minimumDifficulty), bigint(difficulty)) );

		if (gasLimit < c_minGasLimit)
			BOOST_THROW_EXCEPTION(InvalidGasLimit() << RequirementError(bigint(c_minGasLimit), bigint(gasLimit)) );

		if (number && extraData.size() > c_maximumExtraDataSize)
			BOOST_THROW_EXCEPTION(ExtraDataTooBig() << RequirementError(bigint(c_maximumExtraDataSize), bigint(extraData.size())));
	}
}

void BlockInfo::populate(bytesConstRef _block, Strictness _s, h256 const& _h)
{
	RLP root(_block);
	if (!root.isList())
		BOOST_THROW_EXCEPTION(InvalidBlockFormat() << errinfo_comment("block needs to be a list") << BadFieldError(0, _block.toString()));

	RLP header = root[0];

	if (!header.isList())
		BOOST_THROW_EXCEPTION(InvalidBlockFormat() << errinfo_comment("block header needs to be a list") << BadFieldError(0, header.data().toString()));
	populateFromHeader(header, _s, _h);

	if (!root[1].isList())
		BOOST_THROW_EXCEPTION(InvalidBlockFormat() << errinfo_comment("block transactions need to be a list") << BadFieldError(1, root[1].data().toString()));
	if (!root[2].isList())
		BOOST_THROW_EXCEPTION(InvalidBlockFormat() << errinfo_comment("block uncles need to be a list") << BadFieldError(2, root[2].data().toString()));
}

struct BlockInfoDiagnosticsChannel: public LogChannel { static const char* name() { return EthBlue "▧" EthWhite " ◌"; } static const int verbosity = 9; };

void BlockInfo::verifyInternals(bytesConstRef _block) const
{
	RLP root(_block);

	auto txList = root[1];
	auto expectedRoot = trieRootOver(txList.itemCount(), [&](unsigned i){ return rlp(i); }, [&](unsigned i){ return txList[i].data().toBytes(); });

	clog(BlockInfoDiagnosticsChannel) << "Expected trie root:" << toString(expectedRoot);
	if (transactionsRoot != expectedRoot)
	{
		MemoryDB tm;
		GenericTrieDB<MemoryDB> transactionsTrie(&tm);
		transactionsTrie.init();

		vector<bytesConstRef> txs;

		for (unsigned i = 0; i < txList.itemCount(); ++i)
		{
			RLPStream k;
			k << i;

			transactionsTrie.insert(&k.out(), txList[i].data());

			txs.push_back(txList[i].data());
			cdebug << toHex(k.out()) << toHex(txList[i].data());
		}
		cdebug << "trieRootOver" << expectedRoot;
		cdebug << "orderedTrieRoot" << orderedTrieRoot(txs);
		cdebug << "TrieDB" << transactionsTrie.root();
		cdebug << "Contents:";
		for (auto const& t: txs)
			cdebug << toHex(t);

		BOOST_THROW_EXCEPTION(InvalidTransactionsRoot() << Hash256RequirementError(expectedRoot, transactionsRoot));
	}
	clog(BlockInfoDiagnosticsChannel) << "Expected uncle hash:" << toString(sha3(root[2].data()));
	if (sha3Uncles != sha3(root[2].data()))
		BOOST_THROW_EXCEPTION(InvalidUnclesHash());
}

void BlockInfo::populateFromParent(BlockInfo const& _parent)
{
	noteDirty();
	stateRoot = _parent.stateRoot;
	parentHash = _parent.hash();
	number = _parent.number + 1;
	gasLimit = selectGasLimit(_parent);
	gasUsed = 0;
	difficulty = calculateDifficulty(_parent);
}

u256 BlockInfo::selectGasLimit(BlockInfo const& _parent) const
{
	if (!parentHash)
		return c_genesisGasLimit;
	else
		// target minimum of 3141592
		if (_parent.gasLimit < c_genesisGasLimit)
			return min<u256>(c_genesisGasLimit, _parent.gasLimit + _parent.gasLimit / c_gasLimitBoundDivisor - 1);
		else
			return max<u256>(c_genesisGasLimit, _parent.gasLimit - _parent.gasLimit / c_gasLimitBoundDivisor + 1 + (_parent.gasUsed * 6 / 5) / c_gasLimitBoundDivisor);
}

u256 BlockInfo::calculateDifficulty(BlockInfo const& _parent) const
{
	if (!parentHash)
		return (u256)c_genesisDifficulty;
	else
		return max<u256>(c_minimumDifficulty, timestamp >= _parent.timestamp + c_durationLimit ? _parent.difficulty - (_parent.difficulty / c_difficultyBoundDivisor) : (_parent.difficulty + (_parent.difficulty / c_difficultyBoundDivisor)));
}

void BlockInfo::verifyParent(BlockInfo const& _parent) const
{
	// Check difficulty is correct given the two timestamps.
	if (difficulty != calculateDifficulty(_parent))
		BOOST_THROW_EXCEPTION(InvalidDifficulty() << RequirementError((bigint)calculateDifficulty(_parent), (bigint)difficulty));

	if (gasLimit < c_minGasLimit ||
		gasLimit <= _parent.gasLimit - _parent.gasLimit / c_gasLimitBoundDivisor ||
		gasLimit >= _parent.gasLimit + _parent.gasLimit / c_gasLimitBoundDivisor)
		BOOST_THROW_EXCEPTION(InvalidGasLimit() << errinfo_min((bigint)_parent.gasLimit - _parent.gasLimit / c_gasLimitBoundDivisor) << errinfo_got((bigint)gasLimit) << errinfo_max((bigint)_parent.gasLimit + _parent.gasLimit / c_gasLimitBoundDivisor));

	// Check timestamp is after previous timestamp.
	if (parentHash)
	{
		if (parentHash != _parent.hash())
			BOOST_THROW_EXCEPTION(InvalidParentHash());

		if (timestamp <= _parent.timestamp)
			BOOST_THROW_EXCEPTION(InvalidTimestamp());

		if (number != _parent.number + 1)
			BOOST_THROW_EXCEPTION(InvalidNumber());
	}
}
