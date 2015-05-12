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
	size_t headerSize = m_encodedData.size() & ~0x1fUL; //ignore any prefix that is not 32-byte aligned
	//apply offsets
	for (auto const& p: m_offsetMap)
	{
		vector_ref<byte> offsetRef(r.data() + p.first, 32);
		toBigEndian<size_t, vector_ref<byte>>(p.second + headerSize, offsetRef); //add header size minus signature hash
	}

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
	u256 count = 1;
	QStringList strList;
	if (_type.array)
	{
		if (_data.type() == QVariant::String)
			strList = _data.toString().split(",", QString::SkipEmptyParts);  //TODO: proper parsing
		else
			strList = _data.toStringList();
		count = strList.count();

	}
	else
		strList.append(_data.toString());

	if (_type.dynamicSize)
	{
		bytes empty(32);
		size_t sizePos = m_dynamicData.size();
		m_dynamicData += empty; //reserve space for count
		if (_type.type == SolidityType::Type::Bytes)
			count = encodeSingleItem(_data.toString(), _type, m_dynamicData);
		else
		{
			count = strList.count();
			for (auto const& item: strList)
				encodeSingleItem(item, _type, m_dynamicData);
		}
		vector_ref<byte> sizeRef(m_dynamicData.data() + sizePos, 32);
		toBigEndian(count, sizeRef);
		m_offsetMap.push_back(std::make_pair(m_encodedData.size(), sizePos));
		m_encodedData += empty; //reserve space for offset
	}
	else
	{
		if (_type.array)
			count = _type.count;
		int c = static_cast<int>(count);
		if (strList.size() > c)
			strList.erase(strList.begin() + c, strList.end());
		else
			while (strList.size() < c)
				strList.append(QString());

		for (auto const& item: strList)
			encodeSingleItem(item, _type, m_encodedData);
	}
}

unsigned ContractCallDataEncoder::encodeSingleItem(QString const& _data, SolidityType const& _type, bytes& _dest)
{
	if (_type.type == SolidityType::Type::Struct)
		BOOST_THROW_EXCEPTION(dev::Exception() << dev::errinfo_comment("Struct parameters are not supported yet"));

	unsigned const alignSize = 32;
	QString src = _data;
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
		catch (std::exception const&)
		{
			// manage input as a string.
			result = encodeStringParam(src, alignSize);
		}
	}

	size_t dataSize = _type.dynamicSize ? result.size() : alignSize;
	if (result.size() % alignSize != 0)
		result.resize((result.size() & ~(alignSize - 1)) + alignSize);
	_dest.insert(_dest.end(), result.begin(), result.end());
	return dataSize;
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

dev::bytes ContractCallDataEncoder::encodeStringParam(QString const& _str, unsigned alignSize)
{
	bytes result;
	QByteArray bytesAr = _str.toLocal8Bit();
	result = bytes(bytesAr.begin(), bytesAr.end());
	return paddedRight(result, alignSize);
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
	if (type == QSolidityType::Type::SignedInteger || type == QSolidityType::Type::UnsignedInteger)
		return QVariant::fromValue(toString(decodeInt(rawParam)));
	else if (type == QSolidityType::Type::Bool)
		return QVariant::fromValue(toString(decodeBool(rawParam)));
	else if (type == QSolidityType::Type::Bytes || type == QSolidityType::Type::Hash)
		return QVariant::fromValue(toString(decodeBytes(rawParam)));
	else if (type == QSolidityType::Type::Struct)
		return QVariant::fromValue(QString("struct")); //TODO
	else if (type == QSolidityType::Type::Address)
		return QVariant::fromValue(toString(decodeBytes(unpadLeft(rawParam))));
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
