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
/** @file ConstantCompilationModel.h
 * @author Yann yann@ethdev.com
 * @date 2014
 * Ethereum IDE client.
 */

#pragma once

#include <QObject>
#include <libevm/VM.h>
#include <libsolidity/AST.h>

namespace dev
{
namespace mix
{

/**
 * @brief Provides compiler result information.
 */
struct CompilerResult
{
	QString hexCode;
	QString comment;
	dev::bytes bytes;
	bool success;
};

/**
 * @brief Compile source code using the solidity library.
 */
class ConstantCompilationModel
{

public:
	ConstantCompilationModel() {}
	~ConstantCompilationModel() {}
	/// Compile code.
	CompilerResult compile(QString _code);
};

}

}
