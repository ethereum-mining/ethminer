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
/** @file SolidityType.h
* @author Yann yann@ethdev.com
* @author Arkadiy Paronyan arkadiy@ethdev.com
* @date 2015
* Ethereum IDE client.
*/

#pragma once

#include <QString>
#include <vector>
#include <libdevcore/Common.h>

namespace dev
{
namespace mix
{

struct SolidityDeclaration;

//Type info extracted from solidity AST
struct SolidityType
{
	enum Type //keep in sync with QSolidity::Type
	{
		SignedInteger,
		UnsignedInteger,
		Hash, //TODO: remove
		Bool,
		Address,
		Bytes,
		String,
		Enum,
		Struct
	};
	Type type;
	unsigned size; //in bytes,
	unsigned count;
	bool array;
	bool dynamicSize;
	QString name;
	std::vector<SolidityDeclaration> members; //for struct
	std::vector<QString> enumNames; //for enum
	std::shared_ptr<SolidityType const> baseType;
};

struct SolidityDeclaration
{
	QString name;
	SolidityType type;
	dev::u256 slot;
	unsigned offset;
};

using SolidityDeclarations = std::vector<SolidityDeclaration>;

}
}
