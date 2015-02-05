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
/** @file NatspecExpressionEvaluator.h
 * @author Marek Kotewicz <marek@ethdev.com>
 * @date 2015
 */

#include <QtCore/QObject>
#include <QtCore/QtCore>
#include <QtQml/QJSEngine>

class NatspecExpressionEvaluator
{
public:
	NatspecExpressionEvaluator(QString const& _abi = "[]", QString const& _method = "", QString const& _params = "[]");
	
	QString evalExpression(QString const& _expression);
	
private:
	QJSEngine m_engine;
};
