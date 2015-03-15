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
/** @file ContractCallDataEncoder.cpp
 * @author Yann yann@ethdev.com
 * @date 2014
 * Ethereum IDE client.
 */

#include <QDebug>
#include <QMap>
#include <QStringList>
#include <libethcore/CommonJS.h>
#include <libsolidity/AST.h>
#include "QVariableDeclaration.h"
#include "QVariableDefinition.h"
#include "QFunctionDefinition.h"
#include "ContractCallDataEncoder.h"
using namespace dev;
using namespace dev::solidity;
using namespace dev::mix;

bytes ContractCallDataEncoder::encodedData()
{
	return m_encodedData;
}

void ContractCallDataEncoder::encode(QFunctionDefinition const* _function)
{
	bytes hash = _function->hash().asBytes();
	m_encodedData.insert(m_encodedData.end(), hash.begin(), hash.end());
}

void ContractCallDataEncoder::push(bytes const& _b)
{
	m_encodedData.insert(m_encodedData.end(), _b.begin(), _b.end());
}

bigint ContractCallDataEncoder::decodeInt(dev::bytes const& _rawValue)
{
	dev::u256 un = dev::fromBigEndian<dev::u256>(_rawValue);
	if (un >> 255)
		return (-s256(~un + 1));
	return un;
}

dev::bytes ContractCallDataEncoder::encodeInt(QString const& _str)
{
	dev::bigint i(_str.toStdString());
	bytes ret(32);
	toBigEndian((u256)i, ret);
	return ret;
}

QString ContractCallDataEncoder::toString(dev::bigint const& _int)
{
	std::stringstream str;
	str << std::dec << _int;
	return QString::fromStdString(str.str());
}

dev::bytes ContractCallDataEncoder::encodeBool(QString const& _str)
{
	bytes b(1);
	b[0] = _str == "1" || _str.toLower() == "true " ? 1 : 0;
	return padded(b, 32);
}

bool ContractCallDataEncoder::decodeBool(dev::bytes const& _rawValue)
{
	byte ret = _rawValue.at(_rawValue.size() - 1);
	return (ret != 0);
}

QString ContractCallDataEncoder::toString(bool _b)
{
	return _b ? "true" : "false";
}

dev::bytes ContractCallDataEncoder::encodeBytes(QString const& _str)
{
	QByteArray bytesAr = _str.toLocal8Bit();
	bytes r = bytes(bytesAr.begin(), bytesAr.end());
	return padded(r, 32);
}

dev::bytes ContractCallDataEncoder::decodeBytes(dev::bytes const& _rawValue)
{
	return _rawValue;
}

QString ContractCallDataEncoder::toString(dev::bytes const& _b)
{
	return QString::fromStdString(dev::toJS(_b));
}

QStringList ContractCallDataEncoder::decode(QList<QVariableDeclaration*> const& _returnParameters, bytes _value)
{
	bytesConstRef value(&_value);
	bytes rawParam(32);
	QStringList r;
	for (int k = 0; k <_returnParameters.length(); k++)
	{
		value.populate(&rawParam);
		value =  value.cropped(32);
		QVariableDeclaration* dec = static_cast<QVariableDeclaration*>(_returnParameters.at(k));
		QSolidityType::Type type = dec->type()->type();
		if (type == QSolidityType::Type::SignedInteger || type == QSolidityType::Type::UnsignedInteger)
			r.append(toString(decodeInt(rawParam)));
		else if (type == QSolidityType::Type::Bool)
			r.append(toString(decodeBool(rawParam)));
		else if (type == QSolidityType::Type::Bytes || type == QSolidityType::Type::Hash)
			r.append(toString(decodeBytes(rawParam)));
		else if (type == QSolidityType::Type::Struct)
			r.append("struct"); //TODO
		else
			BOOST_THROW_EXCEPTION(Exception() << errinfo_comment("Parameter declaration not found"));
	}
	return r;
}
