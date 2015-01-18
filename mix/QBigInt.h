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
/** @file QBigInt.h
 * @author Yann yann@ethdev.com
 * @date 2015
 * Represent a big integer (u256, bigint, ...) to be used in QML.
 */

#pragma once

#include "boost/variant.hpp"
#include "boost/variant/multivisitors.hpp"
#include <QObject>
#include <libdevcore/CommonJS.h>
#include <libdevcore/Common.h>

using namespace dev;

namespace dev
{
namespace mix
{

using BigIntVariant = boost::variant<dev::u256, dev::bigint>;

struct add: public boost::static_visitor<BigIntVariant>
{
	template<class T1, class T2>
	BigIntVariant operator()(T1 _value, T2 _otherValue) const { return _value + _otherValue; }
};

struct subtract: public boost::static_visitor<BigIntVariant>
{
	template<class T1, class T2>
	BigIntVariant operator()(T1 _value, T2 _otherValue) const { return _value - _otherValue; }
};

struct multiply: public boost::static_visitor<BigIntVariant>
{
	template<class T1, class T2>
	BigIntVariant operator()(T1 _value, T2 _otherValue) const { return _value * _otherValue; }
};

struct divide: public boost::static_visitor<BigIntVariant>
{
	template<class T1, class T2>
	BigIntVariant operator()(T1 _value, T2 _otherValue) const { return _value / _otherValue; }
};

class QBigInt: public QObject
{
	Q_OBJECT

public:
	QBigInt(dev::u256 const& _value, QObject* _parent = 0): QObject(_parent), m_internalValue(_value) {}
	QBigInt(dev::bigint const& _value, QObject* _parent = 0): QObject(_parent), m_internalValue(_value) {}
	QBigInt(BigIntVariant const& _value, QObject* _parent = 0): QObject(_parent), m_internalValue(_value){}
	~QBigInt() {}

	BigIntVariant internalValue() { return m_internalValue; }
	Q_INVOKABLE QString value() const;
	Q_INVOKABLE QBigInt* subtract(QBigInt* const& _value) const;
	Q_INVOKABLE QBigInt* add(QBigInt* const& _value) const;
	Q_INVOKABLE QBigInt* multiply(QBigInt* const& _value) const;
	Q_INVOKABLE QBigInt* divide(QBigInt* const& _value) const;

protected:
	 BigIntVariant m_internalValue;
};

}
}


