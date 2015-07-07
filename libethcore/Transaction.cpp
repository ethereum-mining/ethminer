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
/** @file TransactionBase.cpp
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 */

#include <libdevcore/vector_ref.h>
#include <libdevcore/Log.h>
#include <libdevcore/CommonIO.h>
#include <libdevcrypto/Common.h>
#include <libethcore/Exceptions.h>
#include "Transaction.h"
using namespace std;
using namespace dev;
using namespace dev::eth;

TransactionBase::TransactionBase(TransactionSkeleton const& _ts, Secret const& _s):
	m_type(_ts.creation ? ContractCreation : MessageCall),
	m_nonce(_ts.nonce),
	m_value(_ts.value),
	m_receiveAddress(_ts.to),
	m_gasPrice(_ts.gasPrice),
	m_gas(_ts.gas),
	m_data(_ts.data),
	m_sender(_ts.from)
{
	if (_s)
		sign(_s);
}

TransactionBase::TransactionBase(bytesConstRef _rlpData, CheckTransaction _checkSig)
{
	int field = 0;
	RLP rlp(_rlpData);
	try
	{
		if (!rlp.isList())
			BOOST_THROW_EXCEPTION(InvalidTransactionFormat() << errinfo_comment("transaction RLP must be a list"));

		m_nonce = rlp[field = 0].toInt<u256>();
		m_gasPrice = rlp[field = 1].toInt<u256>();
		m_gas = rlp[field = 2].toInt<u256>();
		m_type = rlp[field = 3].isEmpty() ? ContractCreation : MessageCall;
		m_receiveAddress = rlp[field = 3].isEmpty() ? Address() : rlp[field = 3].toHash<Address>(RLP::VeryStrict);
		m_value = rlp[field = 4].toInt<u256>();

		if (!rlp[field = 5].isData())
			BOOST_THROW_EXCEPTION(InvalidTransactionFormat() << errinfo_comment("transaction data RLP must be an array"));

		m_data = rlp[field = 5].toBytes();
		byte v = rlp[field = 6].toInt<byte>() - 27;
		h256 r = rlp[field = 7].toInt<u256>();
		h256 s = rlp[field = 8].toInt<u256>();

		if (rlp.itemCount() > 9)
			BOOST_THROW_EXCEPTION(InvalidTransactionFormat() << errinfo_comment("to many fields in the transaction RLP"));

		m_vrs = SignatureStruct{ r, s, v };
		if (_checkSig >= CheckTransaction::Cheap && !m_vrs.isValid())
			BOOST_THROW_EXCEPTION(InvalidSignature());
		if (_checkSig == CheckTransaction::Everything)
			m_sender = sender();
	}
	catch (Exception& _e)
	{
		_e << errinfo_name("invalid transaction format") << BadFieldError(field, toHex(rlp[field].data().toBytes()));
		throw;
	}
}

Address const& TransactionBase::safeSender() const noexcept
{
	try
	{
		return sender();
	}
	catch (...)
	{
		cwarn << "safeSender() did throw an exception: " <<  boost::current_exception_diagnostic_information();
		return ZeroAddress;
	}
}

Address const& TransactionBase::sender() const
{
	if (!m_sender)
	{
		auto p = recover(m_vrs, sha3(WithoutSignature));
		if (!p)
			BOOST_THROW_EXCEPTION(InvalidSignature());
		m_sender = right160(dev::sha3(bytesConstRef(p.data(), sizeof(p))));
	}
	return m_sender;
}

void TransactionBase::sign(Secret const& _priv)
{
	auto sig = dev::sign(_priv, sha3(WithoutSignature));
	SignatureStruct sigStruct = *(SignatureStruct const*)&sig;
	if (sigStruct.isValid())
		m_vrs = sigStruct;
}

void TransactionBase::streamRLP(RLPStream& _s, IncludeSignature _sig) const
{
	if (m_type == NullTransaction)
		return;
	_s.appendList((_sig ? 3 : 0) + 6);
	_s << m_nonce << m_gasPrice << m_gas;
	if (m_type == MessageCall)
		_s << m_receiveAddress;
	else
		_s << "";
	_s << m_value << m_data;
	if (_sig)
		_s << (m_vrs.v + 27) << (u256)m_vrs.r << (u256)m_vrs.s;
}
