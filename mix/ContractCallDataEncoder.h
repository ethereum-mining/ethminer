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
/** @file ContractCallDataEncoder.h
 * @author Yann yann@ethdev.com
 * @date 2014
 * Ethereum IDE client.
 */

#pragma once

#include "QVariableDeclaration.h"
#include "QVariableDefinition.h"

namespace dev
{
namespace mix
{

class QFunctionDefinition;
class QVariableDeclaration;
class QVariableDefinition;
class QSolidityType;

/**
 * @brief Encode/Decode data to be sent to a transaction or to be displayed in a view.
 */
class ContractCallDataEncoder
{
public:
	ContractCallDataEncoder() {}
	/// Encode hash of the function to call.
	void encode(QFunctionDefinition const* _function);
	/// Encode data for corresponding type
	void encode(QVariant const& _data, SolidityType const& _type);
	/// Decode variable in order to be sent to QML view.
	QStringList decode(QList<QVariableDeclaration*> const& _dec, bytes _value);
	/// Decode @param _parameter
	QString decode(QVariableDeclaration* const& _param, bytes _value);
	/// Decode single variable
	QVariant decode(SolidityType const& _type, bytes const& _value);
	/// Get all encoded data encoded by encode function.
	bytes encodedData();
	/// Encode a string to bytes (in order to be used as funtion param)
	dev::bytes encodeStringParam(QString const& _str, unsigned _alignSize);
	/// Encode a string to ABI bytes
	dev::bytes encodeBytes(QString const& _str);
	/// Decode bytes from ABI
	dev::bytes decodeBytes(dev::bytes const& _rawValue);
	/// Decode array
	QJsonArray decodeArray(SolidityType const& _type, bytes const& _value, int& pos);
	/// Decode array items
	QJsonValue decodeArrayContent(SolidityType const& _type, bytes const& _value, int& pos);
	/// Decode enum
	QString decodeEnum(bytes _value);

private:
	unsigned encodeSingleItem(QString const& _data, SolidityType const& _type, bytes& _dest);
	bigint decodeInt(dev::bytes const& _rawValue);
	dev::bytes encodeInt(QString const& _str);
	QString toString(dev::bigint const& _int);
	dev::bytes encodeBool(QString const& _str);
	bool decodeBool(dev::bytes const& _rawValue);
	QString toString(bool _b);
	QString toString(dev::bytes const& _b);
	bool asString(dev::bytes const& _b, QString& _str);
	void encodeArray(QJsonArray const& _array, SolidityType const& _type, bytes& _content);
	QString toChar(dev::bytes const& _b);

private:
	bytes m_encodedData;
	bytes m_dynamicData;
	std::vector<std::pair<size_t, size_t>> m_dynamicOffsetMap;
	std::vector<std::pair<size_t, size_t>> m_staticOffsetMap;
};

}
}
