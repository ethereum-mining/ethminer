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
	bytes r(m_encodedData);
	r.insert(r.end(), m_dynamicData.begin(), m_dynamicData.end());
	return r;
}

void ContractCallDataEncoder::encode(QFunctionDefinition const* _function)
{
	bytes hash = _function->hash().asBytes();
	m_encodedData.insert(m_encodedData.end(), hash.begin(), hash.end());
}

void ContractCallDataEncoder::encode(QVariant const& _data, SolidityType const& _type)
{
	if (_type.dynamicSize)
	{
		u256 count = 0;
		if (_type.type == SolidityType::Type::Bytes)
			count = encodeSingleItem(_data, _type, m_dynamicData);
		else
		{
			QVariantList list = qvariant_cast<QVariantList>(_data);
			for (auto const& item: list)
				encodeSingleItem(item, _type, m_dynamicData);
			count = list.size();
		}
		bytes sizeEnc(32);
		toBigEndian(count, sizeEnc);
		m_encodedData.insert(m_encodedData.end(), sizeEnc.begin(), sizeEnc.end());
	}
	else
		encodeSingleItem(_data, _type, m_encodedData);
}

unsigned ContractCallDataEncoder::encodeSingleItem(QVariant const& _data, SolidityType const& _type, bytes& _dest)
{
	if (_type.type == SolidityType::Type::Struct)
		BOOST_THROW_EXCEPTION(dev::Exception() << dev::errinfo_comment("Struct parameters are not supported yet"));

	unsigned const alignSize = 32;
	QString src = _data.toString();
	bytes result;

	if ((src.startsWith("\"") && src.endsWith("\"")) || (src.startsWith("\'") && src.endsWith("\'")))
		src = src.remove(src.length() - 1, 1).remove(0, 1);

	if (src.startsWith("0x"))
	{
		result = fromHex(src.toStdString().substr(2));
		if (_type.type != SolidityType::Type::Bytes)
			result = padded(result, alignSize);
	}
	else
	{
		try
		{
			bigint i(src.toStdString());
			result = bytes(alignSize);
			toBigEndian((u256)i, result);
		}
		catch (std::exception const& ex)
		{
			// manage input as a string.
			QByteArray bytesAr = src.toLocal8Bit();
			result = bytes(bytesAr.begin(), bytesAr.end());
			result = paddedRight(result, alignSize);
		}
	}

	unsigned dataSize = _type.dynamicSize ? result.size() : alignSize;
	_dest.insert(_dest.end(), result.begin(), result.end());
	if ((_dest.size() - 4) % alignSize != 0)
		_dest.resize((_dest.size() & ~(alignSize - 1)) + alignSize);
	return dataSize;
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
	QString str;
	if (asString(_b, str))
		return  "\"" + str +  "\" " + QString::fromStdString(dev::toJS(_b));
	else
		return QString::fromStdString(dev::toJS(_b));
}


QVariant ContractCallDataEncoder::decode(SolidityType const& _type, bytes const& _value)
{
	bytesConstRef value(&_value);
	bytes rawParam(32);
	value.populate(&rawParam);
	QSolidityType::Type type = _type.type;
	if (type == QSolidityType::Type::SignedInteger || type == QSolidityType::Type::UnsignedInteger || type == QSolidityType::Type::Address)
		return QVariant::fromValue(toString(decodeInt(rawParam)));
	else if (type == QSolidityType::Type::Bool)
		return QVariant::fromValue(toString(decodeBool(rawParam)));
	else if (type == QSolidityType::Type::Bytes || type == QSolidityType::Type::Hash)
		return QVariant::fromValue(toString(decodeBytes(rawParam)));
	else if (type == QSolidityType::Type::Struct)
		return QVariant::fromValue(QString("struct")); //TODO
	else
		BOOST_THROW_EXCEPTION(Exception() << errinfo_comment("Parameter declaration not found"));
}

QStringList ContractCallDataEncoder::decode(QList<QVariableDeclaration*> const& _returnParameters, bytes _value)
{
	bytesConstRef value(&_value);
	bytes rawParam(32);
	QStringList r;

	for (int k = 0; k <_returnParameters.length(); k++)
	{
		value.populate(&rawParam);
		value = value.cropped(32);
		QVariableDeclaration* dec = static_cast<QVariableDeclaration*>(_returnParameters.at(k));
		SolidityType const& type = dec->type()->type();
		r.append(decode(type, rawParam).toString());
	}
	return r;
}


bool ContractCallDataEncoder::asString(dev::bytes const& _b, QString& _str)
{
	dev::bytes bunPad = unpadded(_b);
	for (unsigned i = 0; i < bunPad.size(); i++)
	{
		if (bunPad.at(i) < 9 || bunPad.at(i) > 127)
			return false;
		else
			_str += QString::fromStdString(dev::toJS(bunPad.at(i))).replace("0x", "");
	}
	return true;
}
