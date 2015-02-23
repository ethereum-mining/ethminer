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

#pragma once

#include <QtCore/QObject>
#include <QtCore/QtCore>
#include <QtQml/QJSEngine>

/**
 * Should be used to evaluate natspec expression.
 * @see test/natspec.cpp for natspec expression examples
 */
class NatspecExpressionEvaluator
{
public:
	/// Construct natspec expression evaluator
	/// @params abi - contract's abi in json format, passed as string
	/// @params method - name of the contract's method for which we evaluate the natspec.
	/// If we want to use raw string, it should be passed with quotation marks eg. "\"helloWorld\""
	/// If we pass string "helloWorld", the value of the object with name "helloWorld" will be used
	/// @params params - array of method input params, passed as string, objects in array should be
	/// javascript valid objects
	NatspecExpressionEvaluator(QString const& _abi = "[]", QString const& _method = "", QString const& _params = "[]");
	
	/// Should be called to evaluate natspec expression
	/// @params expression - natspec expression
	/// @returns evaluated natspec expression if it was valid, otherwise original expression
	QString evalExpression(QString const& _expression);
	
private:
	QJSEngine m_engine;
};
