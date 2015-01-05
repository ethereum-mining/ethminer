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
/** @file ConstantCompilationModel.cpp
 * @author Yann yann@ethdev.com
 * @date 2014
 * Ethereum IDE client.
 */

#include <QApplication>
#include <QObject>
#include <libevm/VM.h>
#include <libsolidity/Scanner.h>
#include <libsolidity/Parser.h>
#include <libsolidity/CompilerStack.h>
#include <libsolidity/SourceReferenceFormatter.h>
#include <libsolidity/NameAndTypeResolver.h>
#include "ConstantCompilationModel.h"
using namespace std;
using namespace dev;
using namespace dev::eth;
using namespace dev::mix;
using namespace dev::solidity;

CompilerResult ConstantCompilationModel::compile(QString _code)
{
	dev::solidity::CompilerStack compiler;
	dev::bytes m_data;
	CompilerResult res;
	try
	{
		m_data = compiler.compile(_code.toStdString(), true);
		res.success = true;
		res.comment = "ok";
		res.hexCode = QString::fromStdString(dev::eth::disassemble(m_data));
		res.bytes = m_data;
	}
	catch (dev::Exception const& _exception)
	{
		ostringstream error;
		solidity::SourceReferenceFormatter::printExceptionInformation(error, _exception, "Error", compiler);
		res.success = false;
		res.comment = QString::fromStdString(error.str());
		res.hexCode = "";
	}
	catch (...)
	{
		res.success = false;
		res.comment = QApplication::tr("Uncaught exception.");
		res.hexCode = "";
	}
	return res;
}
