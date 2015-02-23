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
/** @file NatspecExpressionEvaluator.cpp
 * @author Marek Kotewicz <marek@ethdev.com>
 * @date 2015
 */

#include <libdevcore/Log.h>
#include <libdevcore/Exceptions.h>
#include "NatspecExpressionEvaluator.h"

using namespace std;
using namespace dev;

static QString contentsOfQResource(string const& _res)
{
	QFile file(QString::fromStdString(_res));
	if (!file.open(QFile::ReadOnly))
		BOOST_THROW_EXCEPTION(FileError());
	QTextStream in(&file);
	return in.readAll();
}

NatspecExpressionEvaluator::NatspecExpressionEvaluator(QString const& _abi, QString const& _method, QString const& _params)
{
	Q_INIT_RESOURCE(natspec);
	QJSValue result = m_engine.evaluate(contentsOfQResource(":/natspec/natspec.js"));
	if (result.isError())
		BOOST_THROW_EXCEPTION(FileError());

	m_engine.evaluate("globals.abi = " + _abi);
	m_engine.evaluate("globals.method = " + _method);
	m_engine.evaluate("globals.params = " + _params);
}

QString NatspecExpressionEvaluator::evalExpression(QString const& _expression)
{
	QJSValue result = m_engine.evaluate("evaluateExpression(\"" + _expression + "\")");
	if (result.isError())
	{
		cerr << "Could not evaluate expression: \"" << _expression.toStdString() << "\"" << endl;
		return _expression;
	}
	return result.toString();
}
