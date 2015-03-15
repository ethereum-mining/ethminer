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

namespace dev
{
namespace mix
{

struct SolidityDeclaration;

struct SolidityType
{
	enum Type //TODO: arrays and structs
	{
		SignedInteger,
		UnsignedInteger,
		Hash,
		Bool,
		Address,
		String,
		Enum,
		Struct
	};
	Type type;
	unsigned size; //bytes,
	bool array;
	bool dynamicSize;
	QString name;
	std::vector<SolidityDeclaration> members;
	std::vector<QString> enumNames;
};

struct SolidityDeclaration
{
	QString name;
	SolidityType type;
};

}
}
