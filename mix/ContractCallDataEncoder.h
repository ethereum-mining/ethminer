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

/**
 * @brief Encode/Decode data to be sent to a transaction or to be displayed in a view.
 */
class ContractCallDataEncoder
{
public:
	ContractCallDataEncoder() {}
	/// Encode hash of the function to call.
	void encode(QFunctionDefinition const* _function);
	/// Decode variable in order to be sent to QML view.
	QList<QVariableDefinition*> decode(QList<QVariableDeclaration*> const& _dec, bytes _value);
	/// Get all encoded data encoded by encode function.
	bytes encodedData();
	/// Push the given @a _b to the current param context.
	void push(bytes const& _b);

private:
	bytes m_encodedData;
};

}
}
