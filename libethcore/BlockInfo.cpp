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

BlockInfo::BlockInfo(): m_timestamp(Invalid256)
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
	m_parentHash = h256();
	m_sha3Uncles = EmptyListSHA3;
	m_coinbaseAddress = Address();
	m_stateRoot = EmptyTrie;
	m_transactionsRoot = EmptyTrie;
	m_receiptsRoot = EmptyTrie;
	m_logBloom = LogBloom();
	m_difficulty = 0;
	m_number = 0;
	m_gasLimit = 0;
	m_gasUsed = 0;
	m_timestamp = 0;
	m_extraData.clear();
	noteDirty();
}

h256 const& BlockInfo::boundary() const
{
	if (!m_boundary && m_difficulty)
		m_boundary = (h256)(u256)((bigint(1) << 256) / m_difficulty);
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
	_s	<< m_parentHash << m_sha3Uncles << m_coinbaseAddress << m_stateRoot << m_transactionsRoot << m_receiptsRoot << m_logBloom
		<< m_difficulty << m_number << m_gasLimit << m_gasUsed << m_timestamp << m_extraData;
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
		m_parentHash = _header[field = 0].toHash<h256>(RLP::VeryStrict);
		m_sha3Uncles = _header[field = 1].toHash<h256>(RLP::VeryStrict);
		m_coinbaseAddress = _header[field = 2].toHash<Address>(RLP::VeryStrict);
		m_stateRoot = _header[field = 3].toHash<h256>(RLP::VeryStrict);
		m_transactionsRoot = _header[field = 4].toHash<h256>(RLP::VeryStrict);
		m_receiptsRoot = _header[field = 5].toHash<h256>(RLP::VeryStrict);
		m_logBloom = _header[field = 6].toHash<LogBloom>(RLP::VeryStrict);
		m_difficulty = _header[field = 7].toInt<u256>();
		m_number = _header[field = 8].toInt<u256>();
		m_gasLimit = _header[field = 9].toInt<u256>();
		m_gasUsed = _header[field = 10].toInt<u256>();
		m_timestamp = _header[field = 11].toInt<u256>();
		m_extraData = _header[field = 12].toBytes();
	}
	catch (Exception const& _e)
	{
		_e << errinfo_name("invalid block header format") << BadFieldError(field, toHex(_header[field].data().toBytes()));
		throw;
	}

	if (m_number > ~(unsigned)0)
		BOOST_THROW_EXCEPTION(InvalidNumber());

	if (_s != CheckNothing && m_gasUsed > m_gasLimit)
		BOOST_THROW_EXCEPTION(TooMuchGasUsed() << RequirementError(bigint(m_gasLimit), bigint(m_gasUsed)));
}

struct BlockInfoDiagnosticsChannel: public LogChannel { static const char* name() { return EthBlue "▧" EthWhite " ◌"; } static const int verbosity = 9; };

void BlockInfo::verifyInternals(bytesConstRef _block) const
{
	RLP root(_block);

	auto txList = root[1];
	auto expectedRoot = trieRootOver(txList.itemCount(), [&](unsigned i){ return rlp(i); }, [&](unsigned i){ return txList[i].data().toBytes(); });

	clog(BlockInfoDiagnosticsChannel) << "Expected trie root:" << toString(expectedRoot);
	if (m_transactionsRoot != expectedRoot)
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

		BOOST_THROW_EXCEPTION(InvalidTransactionsRoot() << Hash256RequirementError(expectedRoot, m_transactionsRoot));
	}
	clog(BlockInfoDiagnosticsChannel) << "Expected uncle hash:" << toString(sha3(root[2].data()));
	if (m_sha3Uncles != sha3(root[2].data()))
		BOOST_THROW_EXCEPTION(InvalidUnclesHash() << Hash256RequirementError(sha3(root[2].data()), m_sha3Uncles));
}

void BlockInfo::populateFromParent(BlockInfo const& _parent)
{
	m_stateRoot = _parent.stateRoot();
	m_number = _parent.m_number + 1;
	m_parentHash = _parent.m_hash;
	m_gasLimit = _parent.childGasLimit();
	m_gasUsed = 0;
	m_difficulty = calculateDifficulty(_parent);
}

u256 BlockInfo::childGasLimit(u256 const& _gasFloorTarget) const
{
	u256 gasFloorTarget =
		_gasFloorTarget == UndefinedU256 ? c_gasFloorTarget : _gasFloorTarget;

	if (m_gasLimit < gasFloorTarget)
		return min<u256>(gasFloorTarget, m_gasLimit + m_gasLimit / c_gasLimitBoundDivisor - 1);
	else
		return max<u256>(gasFloorTarget, m_gasLimit - m_gasLimit / c_gasLimitBoundDivisor + 1 + (m_gasUsed * 6 / 5) / c_gasLimitBoundDivisor);
}

u256 BlockInfo::calculateDifficulty(BlockInfo const& _parent) const
{
	const unsigned c_expDiffPeriod = 100000;

	if (!m_number)
		throw GenesisBlockCannotBeCalculated();
	u256 o = max<u256>(c_minimumDifficulty, m_timestamp >= _parent.m_timestamp + c_durationLimit ? _parent.m_difficulty - (_parent.m_difficulty / c_difficultyBoundDivisor) : (_parent.m_difficulty + (_parent.m_difficulty / c_difficultyBoundDivisor)));
	unsigned periodCount = unsigned(_parent.number() + 1) / c_expDiffPeriod;
	if (periodCount > 1)
		o = max<u256>(c_minimumDifficulty, o + (u256(1) << (periodCount - 2)));	// latter will eventually become huge, so ensure it's a bigint.
	return o;
}

void BlockInfo::verifyParent(BlockInfo const& _parent) const
{
	// Check timestamp is after previous timestamp.
	if (m_parentHash)
	{
		if (m_timestamp <= _parent.m_timestamp)
			BOOST_THROW_EXCEPTION(InvalidTimestamp());

		if (m_number != _parent.m_number + 1)
			BOOST_THROW_EXCEPTION(InvalidNumber());
	}
}
