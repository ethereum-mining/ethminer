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

#include "BlockHeader.h"
#include <libdevcore/Common.h>
#include <libdevcore/Log.h>
#include <libdevcore/RLP.h>
#include "EthashAux.h"

using namespace std;
using namespace dev;
using namespace dev::eth;


BlockHeader::BlockHeader(bytesConstRef _block)
{
	RLP header = extractHeader(_block);
	populateFromHeader(header);
}

h256 const& BlockHeader::boundary() const
{
	if (!m_boundary && m_difficulty)
		m_boundary = (h256)(u256)((bigint(1) << 256) / m_difficulty);
	return m_boundary;
}

h256 const& BlockHeader::hashWithout() const
{
	if (!m_hashWithout)
	{
		RLPStream s(BasicFields);
		streamRLPFields(s);
		m_hashWithout = sha3(s.out());
	}
	return m_hashWithout;
}

void BlockHeader::streamRLPFields(RLPStream& _s) const
{
	_s	<< m_parentHash << m_sha3Uncles << m_coinbaseAddress << m_stateRoot << m_transactionsRoot << m_receiptsRoot << m_logBloom
		<< m_difficulty << m_number << m_gasLimit << m_gasUsed << m_timestamp << m_extraData;
}

RLP BlockHeader::extractHeader(bytesConstRef _block)
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

void BlockHeader::populateFromHeader(RLP const& _header)
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

	if (m_gasUsed > m_gasLimit)
		BOOST_THROW_EXCEPTION(TooMuchGasUsed() << RequirementError(bigint(m_gasLimit), bigint(m_gasUsed)));
}
