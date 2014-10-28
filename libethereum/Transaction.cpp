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
/** @file Transaction.cpp
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 */

#include <libdevcore/vector_ref.h>
#include <libdevcore/Log.h>
#include <libdevcrypto/Common.h>
#include <libethcore/Exceptions.h>
#include "Transaction.h"
using namespace std;
using namespace dev;
using namespace dev::eth;

#define ETH_ADDRESS_DEBUG 0

Transaction::Transaction(bytesConstRef _rlpData, bool _checkSender)
{
	int field = 0;
	RLP rlp(_rlpData);
	try
	{
		nonce = rlp[field = 0].toInt<u256>();
		gasPrice = rlp[field = 1].toInt<u256>();
		gas = rlp[field = 2].toInt<u256>();
		type = rlp[field = 3].isEmpty() ? ContractCreation : MessageCall;
		receiveAddress = rlp[field = 3].toHash<Address>();
		value = rlp[field = 4].toInt<u256>();
		data = rlp[field = 5].toBytes();
		vrs = SignatureStruct{ rlp[field = 7].toInt<u256>(), rlp[field = 8].toInt<u256>(), byte(rlp[field = 6].toInt<byte>() - 27) };
		if (_checkSender)
			m_sender = sender();
	}
	catch (Exception & _e)
	{
		_e << errinfo_name("invalid transaction format") << BadFieldError(field,toHex(rlp[field].data().toBytes()));
		throw;
	}
}

Address Transaction::safeSender() const noexcept
{
	try
	{
		return sender();
	}
	catch (...)
	{
		cwarn << "safeSender() did throw an exception: " <<  boost::current_exception_diagnostic_information();
		return Address();
	}
}

Address Transaction::sender() const
{
	if (!m_sender)
	{
		auto p = recover(*(Signature const*)&vrs, sha3(false));
		if (!p)
			BOOST_THROW_EXCEPTION(InvalidSignature());
		m_sender = right160(dev::sha3(bytesConstRef(p.data(), sizeof(p))));
	}
	return m_sender;
}

void Transaction::sign(Secret _priv)
{
	auto sig = dev::sign(_priv, sha3(false));
	vrs = *(SignatureStruct const*)&sig;
}

void Transaction::streamRLP(RLPStream& _s, bool _sig) const
{
	_s.appendList((_sig ? 3 : 0) + 6);
	_s << nonce << gasPrice << gas;
	if (type == MessageCall)
		_s << receiveAddress;
	else
		_s << "";
	_s << value << data;
	if (_sig)
		_s << (vrs.v + 27) << (u256)vrs.r << (u256)vrs.s;
}
