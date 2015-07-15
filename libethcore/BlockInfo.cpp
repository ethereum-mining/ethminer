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
#include "Exceptions.h"
#include "BlockInfo.h"
using namespace std;
using namespace dev;
using namespace dev::eth;

BlockInfo::BlockInfo(): timestamp(Invalid256)
{
}

BlockInfo::BlockInfo(bytesConstRef _block, Strictness _s, h256 const& _hashWith, BlockDataType _bdt)
{
	RLP header = _bdt == BlockData ? extractHeader(_block) : RLP(_block);
	m_hash = _hashWith ? _hashWith : sha3(header.data());
	populateFromHeader(header, _s);
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
	noteDirty();
}

h256 const& BlockInfo::boundary() const
{
	if (!m_boundary && difficulty)
		m_boundary = (h256)(u256)((bigint(1) << 256) / difficulty);
	return m_boundary;
}

h256 const& BlockInfo::hashWithout() const
{
	if (!m_hashWithout)
	{
		RLPStream s(BasicFields);
		streamRLPFields(s);
		m_hashWithout = sha3(s.out());
	}
	return m_hashWithout;
}

void BlockInfo::streamRLPFields(RLPStream& _s) const
{
	_s	<< parentHash << sha3Uncles << coinbaseAddress << stateRoot << transactionsRoot << receiptsRoot << logBloom
		<< difficulty << number << gasLimit << gasUsed << timestamp << extraData;
}

h256 BlockInfo::headerHashFromBlock(bytesConstRef _block)
{
	return sha3(RLP(_block)[0].data());
}

RLP BlockInfo::extractHeader(bytesConstRef _block)
{
	RLP root(_block);
	if (!root.isList())
		BOOST_THROW_EXCEPTION(InvalidBlockFormat() << errinfo_comment("block needs to be a list") << BadFieldError(0, _block.toString()));
	RLP header = root[0];
	if (!header.isList())
		BOOST_THROW_EXCEPTION(InvalidBlockFormat() << errinfo_comment("block header needs to be a list") << BadFieldError(0, header.data().toString()));
	if (!root[1].isList())
		BOOST_THROW_EXCEPTION(InvalidBlockFormat() << errinfo_comment("block transactions need to be a list") << BadFieldError(1, root[1].data().toString()));
	if (!root[2].isList())
		BOOST_THROW_EXCEPTION(InvalidBlockFormat() << errinfo_comment("block uncles need to be a list") << BadFieldError(2, root[2].data().toString()));
	return header;
}

void BlockInfo::populateFromHeader(RLP const& _header, Strictness _s)
{
	int field = 0;
	try
	{
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
	}
	catch (Exception const& _e)
	{
		_e << errinfo_name("invalid block header format") << BadFieldError(field, toHex(_header[field].data().toBytes()));
		throw;
	}

	if (number > ~(unsigned)0)
		BOOST_THROW_EXCEPTION(InvalidNumber());

	if (_s != CheckNothing && gasUsed > gasLimit)
		BOOST_THROW_EXCEPTION(TooMuchGasUsed() << RequirementError(bigint(gasLimit), bigint(gasUsed)) );
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
	stateRoot = _parent.stateRoot;
	number = _parent.number + 1;
	gasLimit = selectGasLimit(_parent);
	gasUsed = 0;
	difficulty = calculateDifficulty(_parent);
	parentHash = _parent.hash();
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
	// Check timestamp is after previous timestamp.
	if (parentHash)
	{
		if (timestamp <= _parent.timestamp)
			BOOST_THROW_EXCEPTION(InvalidTimestamp());

		if (number != _parent.number + 1)
			BOOST_THROW_EXCEPTION(InvalidNumber());
	}
}
