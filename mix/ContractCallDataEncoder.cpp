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
#include <libdevcore/CommonJS.h>
#include <libsolidity/AST.h>
#include "QVariableDeclaration.h"
#include "QVariableDefinition.h"
#include "ContractCallDataEncoder.h"
using namespace dev;
using namespace dev::solidity;
using namespace dev::mix;

bytes ContractCallDataEncoder::encodedData()
{
	return m_encodedData;
}

void ContractCallDataEncoder::encode(int _functionIndex)
{
	bytes i = jsToBytes(std::to_string(_functionIndex));
	m_encodedData.insert(m_encodedData.end(), i.begin(), i.end());
}

void ContractCallDataEncoder::encode(QVariableDeclaration* _dec, bool _value)
{
	return encode(_dec, QString(formatBool(_value)));
}

void ContractCallDataEncoder::encode(QVariableDeclaration* _dec, QString _value)
{
	int padding = this->padding(_dec->type());
	bytes data = padded(jsToBytes(_value.toStdString()), padding);
	m_encodedData.insert(m_encodedData.end(), data.begin(), data.end());
}

void ContractCallDataEncoder::encode(QVariableDeclaration* _dec, u256 _value)
{
	int padding = this->padding(_dec->type());
	std::ostringstream s;
	s << std::hex << "0x" << _value;
	bytes data = padded(jsToBytes(s.str()), padding);
	m_encodedData.insert(m_encodedData.end(), data.begin(), data.end());
	encodedData();
}

QList<QVariableDefinition*> ContractCallDataEncoder::decode(QList<QVariableDeclaration*> _returnParameters, bytes _value)
{
	QList<QVariableDefinition*> r;
	std::string returnValue = toJS(_value);
	returnValue = returnValue.substr(2, returnValue.length() - 1);
	for (int k = 0; k <_returnParameters.length(); k++)
	{
		QVariableDeclaration* dec = (QVariableDeclaration*)_returnParameters.at(k);
		int padding = this->padding(dec->type());
		std::string rawParam = returnValue.substr(0, padding * 2);
		r.append(new QVariableDefinition(dec, convertToReadable(unpadLeft(rawParam), dec)));
		returnValue = returnValue.substr(rawParam.length(), returnValue.length() - 1);
	}
	return r;
}

int ContractCallDataEncoder::padding(QString type)
{
	// TODO : to be improved (load types automatically from solidity library).
	if (type.indexOf("uint") != -1)
		return integerPadding(type.remove("uint").toInt());
	else if (type.indexOf("int") != -1)
		return integerPadding(type.remove("int").toInt());
	else if (type.indexOf("bool") != -1)
		return 1;
	else if ((type.indexOf("address") != -1))
		return 20;
	else
		return 0;
}

int ContractCallDataEncoder::integerPadding(int bitValue)
{
	return bitValue / 8;
}

QString ContractCallDataEncoder::formatBool(bool _value)
{
	return (_value ? "1" : "0");
}

QString ContractCallDataEncoder::convertToReadable(std::string _v, QVariableDeclaration* _dec)
{
	if (_dec->type().indexOf("int") != -1)
		return convertToInt(_v);
	else if (_dec->type().indexOf("bool") != -1)
		return convertToBool(_v);
	else
		return QString::fromStdString(_v);
}

QString ContractCallDataEncoder::convertToBool(std::string _v)
{
	return _v == "1" ? "true" : "false";
}

QString ContractCallDataEncoder::convertToInt(std::string _v)
{
	//TO DO to be improve to manage all int, uint size (128, 256, ...) in ethereum QML types task #612.
	int x = std::stol(_v, nullptr, 16);
	std::stringstream ss;
	ss << std::dec << x;
	return QString::fromStdString(ss.str());
}
