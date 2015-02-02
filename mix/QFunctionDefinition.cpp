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
/** @file QFunctionDefinition.cpp
 * @author Yann yann@ethdev.com
 * @date 2014
 */

#include <libsolidity/AST.h>
#include <libdevcrypto/SHA3.h>
#include <libdevcore/Exceptions.h>
#include "QVariableDeclaration.h"
#include "QFunctionDefinition.h"

using namespace dev::solidity;
using namespace dev::mix;

QFunctionDefinition::QFunctionDefinition(dev::solidity::FunctionTypePointer const& _f): QBasicNodeDefinition(&_f->getDeclaration()), m_hash(dev::sha3(_f->getCanonicalSignature()))
{
	auto paramNames = _f->getParameterNames();
	auto paramTypes = _f->getParameterTypeNames();
	auto returnNames = _f->getReturnParameterNames();
	auto returnTypes = _f->getReturnParameterTypeNames();
	for (unsigned i = 0; i < paramNames.size(); ++i)
		m_parameters.append(new QVariableDeclaration(paramNames[i], paramTypes[i]));

	for (unsigned i = 0; i < returnNames.size(); ++i)
		m_returnParameters.append(new QVariableDeclaration(returnNames[i], returnTypes[i]));
}
