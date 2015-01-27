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

void ContractCallDataEncoder::push(bytes _b)
{
	m_encodedData.insert(m_encodedData.end(), _b.begin(), _b.end());
}

QList<QVariableDefinition*> ContractCallDataEncoder::decode(QList<QVariableDeclaration*> _returnParameters, bytes _value)
{
	QList<QVariableDefinition*> r;
	std::string returnValue = toJS(_value);
	returnValue = returnValue.substr(2, returnValue.length() - 1);
	for (int k = 0; k <_returnParameters.length(); k++)
	{
		QVariableDeclaration* dec = (QVariableDeclaration*)_returnParameters.at(k);
		QVariableDefinition* def = nullptr;
		if (dec->type().contains("int"))
			def = new QIntType(dec, QString());
		else if (dec->type().contains("real"))
			def = new QRealType(dec, QString());
		else if (dec->type().contains("string") || dec->type().contains("text"))
			def = new QStringType(dec, QString());
		else if (dec->type().contains("hash") || dec->type().contains("address"))
			def = new QHashType(dec, QString());

		def->decodeValue(returnValue);
		r.push_back(def);
		returnValue = returnValue.substr(def->length(), returnValue.length() - 1);

		/*QStringList tLength = typeLength(dec->type());

		QRegExp intTest("(uint|int|hash|address)");
		QRegExp stringTest("(string|text)");
		QRegExp realTest("(real|ureal)");
		if (intTest.indexIn(dec->type()) != -1)
		{
			std::string rawParam = returnValue.substr(0, (tLength.first().toInt() / 8) * 2);
			QString value = resolveNumber(QString::fromStdString(rawParam));
			r.append(new QVariableDefinition(dec, value));
			returnValue = returnValue.substr(rawParam.length(), returnValue.length() - 1);
		}
		else if (dec->type() == "bool")
		{
			std::string rawParam = returnValue.substr(0, 2);
			std::string unpadded = unpadLeft(rawParam);
			r.append(new QVariableDefinition(dec, QString::fromStdString(unpadded)));
			returnValue = returnValue.substr(rawParam.length(), returnValue.length() - 1);
		}
		else if (stringTest.indexIn(dec->type()) != -1)
		{
			if (tLength.length() == 0)
			{
				QString strLength = QString::fromStdString(returnValue.substr(0, 2));
				returnValue = returnValue.substr(2, returnValue.length() - 1);
				QString strValue = QString::fromStdString(returnValue.substr(0, strLength.toInt()));
				r.append(new QVariableDefinition(dec, strValue));
				returnValue = returnValue.substr(strValue.length(), returnValue.length() - 1);
			}
			else
			{
				std::string rawParam = returnValue.substr(0, (tLength.first().toInt() / 8) * 2);
				r.append(new QVariableDefinition(dec, QString::fromStdString(rawParam)));
				returnValue = returnValue.substr(rawParam.length(), returnValue.length() - 1);
			}
		}
		else if (realTest.indexIn(dec->type()) != -1)
		{
			QString value;
			for (QString str: tLength)
			{
				std::string rawParam = returnValue.substr(0, (str.toInt() / 8) * 2);
				QString value = resolveNumber(QString::fromStdString(rawParam));
				value += value + "x";
				returnValue = returnValue.substr(rawParam.length(), returnValue.length() - 1);
			}
			r.append(new QVariableDefinition(dec, value));
		}*/
	}
	return r;
}

QString ContractCallDataEncoder::resolveNumber(QString const& _rawParam)
{
	std::string unPadded = unpadLeft(_rawParam.toStdString());
	int x = std::stol(unPadded, nullptr, 16);
	std::stringstream ss;
	ss << std::dec << x;
	return QString::fromStdString(ss.str());
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
