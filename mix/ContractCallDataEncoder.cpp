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

QList<QVariableDefinition*> ContractCallDataEncoder::decode(QList<QVariableDeclaration*> const& _returnParameters, bytes _value)
{
	bytesConstRef value(&_value);
	bytes rawParam(32);
	QList<QVariableDefinition*> r;
	for (int k = 0; k <_returnParameters.length(); k++)
	{
		QVariableDeclaration* dec = (QVariableDeclaration*)_returnParameters.at(k);
		QVariableDefinition* def = nullptr;
		if (dec->type().contains("int"))
			def = new QIntType(dec, QString());
		else if (dec->type().contains("real"))
			def = new QRealType(dec, QString());
		else if (dec->type().contains("bool"))
			def = new QBoolType(dec, QString());
		else if (dec->type().contains("string") || dec->type().contains("text"))
			def = new QStringType(dec, QString());
		else if (dec->type().contains("hash") || dec->type().contains("address"))
			def = new QHashType(dec, QString());
		else
			BOOST_THROW_EXCEPTION(Exception() << errinfo_comment("Parameter declaration not found"));

		value.populate(&rawParam);
		def->decodeValue(rawParam);
		r.push_back(def);
		value =  value.cropped(32);
		qDebug() << "decoded return value : " << dec->type() << " " << def->value();
	}
	return r;
}
