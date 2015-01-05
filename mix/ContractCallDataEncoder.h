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

/**
 * @brief Encode/Decode data to be sent to a transaction or to be displayed in a view.
 */
class ContractCallDataEncoder
{
public:
	ContractCallDataEncoder() {}
	/// Encode variable in order to be sent as parameter.
	void encode(QVariableDeclaration* _dec, QString _value);
	/// Encode variable in order to be sent as parameter.
	void encode(QVariableDeclaration* _dec, u256 _value);
	/// Encode variable in order to be sent as parameter.
	void encode(QVariableDeclaration* _dec, bool _value);
	/// Encode index of the function to call.
	void encode(int _functionIndex);
	/// Decode variable in order to be sent to QML view.
	QList<QVariableDefinition*> decode(QList<QVariableDeclaration*> _dec, bytes _value);
	/// Get all encoded data encoded by encode function.
	bytes encodedData();

private:
	int padding(QString _type);
	bytes m_encodedData;
	static QString convertToReadable(std::string _v, QVariableDeclaration* _dec);
	static QString convertToBool(std::string _v);
	static QString convertToInt(std::string _v);
	static int integerPadding(int _bitValue);
	static QString formatBool(bool _value);
};

}
}
