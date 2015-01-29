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

QFunctionDefinition::QFunctionDefinition(dev::solidity::FunctionDescription const& _f): QBasicNodeDefinition(_f.getDeclaration()), m_hash()
{

	FunctionDefinition const* funcDef;
	VariableDeclaration const* varDecl;

	if ((funcDef = _f.getFunctionDefinition()))
	{
		m_hash = FixedHash<4>(dev::sha3(funcDef->getCanonicalSignature()));
		std::vector<std::shared_ptr<VariableDeclaration>> parameters = funcDef->getParameterList().getParameters();
		for (unsigned i = 0; i < parameters.size(); i++)
			m_parameters.append(new QVariableDeclaration(parameters.at(i).get()));

		std::vector<std::shared_ptr<VariableDeclaration>> returnParameters = funcDef->getReturnParameters();
		for (unsigned i = 0; i < returnParameters.size(); i++)
			m_returnParameters.append(new QVariableDeclaration(returnParameters.at(i).get()));
	}
	else
	{
		if (!(varDecl = _f.getVariableDeclaration()))
			BOOST_THROW_EXCEPTION(Exception() << errinfo_comment("Malformed FunctionDescription. Should never happen."));

		// only the return parameter for now.
		// TODO: change this for other state variables like mapping and maybe abstract this inside solidity and not here
		auto returnParams = _f.getReturnParameters();
		m_returnParameters.append(new QVariableDeclaration(returnParams[0]));

	}
}
