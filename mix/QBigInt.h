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
 * Represent a big integer (u256, bigint) to be used in QML.
 */

#pragma once

#include "boost/variant.hpp"
#include "boost/variant/multivisitors.hpp"
#include <QObject>
#include <QQmlEngine>
#include <libethcore/CommonJS.h>
#include <libdevcore/Common.h>

using namespace dev;

namespace dev
{
namespace mix
{

using BigIntVariant = boost::variant<dev::u256, dev::bigint, dev::s256>;

struct add: public boost::static_visitor<BigIntVariant>
{
	template<class T1, class T2>
	BigIntVariant operator()(T1 const& _value, T2 const& _otherValue) const { return _value + _otherValue; }
};

struct subtract: public boost::static_visitor<BigIntVariant>
{
	template<class T1, class T2>
	BigIntVariant operator()(T1 const& _value, T2 const& _otherValue) const { return _value - _otherValue; }
};

struct multiply: public boost::static_visitor<BigIntVariant>
{
	template<class T1, class T2>
	BigIntVariant operator()(T1 const& _value, T2 const& _otherValue) const { return _value * _otherValue; }
};

struct divide: public boost::static_visitor<BigIntVariant>
{
	template<class T1, class T2>
	BigIntVariant operator()(T1 const& _value, T2 const& _otherValue) const { return _value / _otherValue; }
};

/*
 * Represent big integer like big int and u256 in QML.
 * The ownership is set by default to Javascript.
 */
class QBigInt: public QObject
{
	Q_OBJECT

public:
	QBigInt(QObject* _parent = 0): QObject(_parent), m_internalValue(dev::u256(0)) { QQmlEngine::setObjectOwnership(this, QQmlEngine::JavaScriptOwnership); }
	QBigInt(dev::u256 const& _value, QObject* _parent = 0): QObject(_parent), m_internalValue(_value) { QQmlEngine::setObjectOwnership(this, QQmlEngine::JavaScriptOwnership); }
	QBigInt(dev::bigint const& _value, QObject* _parent = 0): QObject(_parent), m_internalValue(_value) { QQmlEngine::setObjectOwnership(this, QQmlEngine::JavaScriptOwnership); }
	QBigInt(BigIntVariant const& _value, QObject* _parent = 0): QObject(_parent), m_internalValue(_value){ QQmlEngine::setObjectOwnership(this, QQmlEngine::JavaScriptOwnership); }
	QBigInt(dev::s256 const& _value, QObject* _parent = 0): QObject(_parent), m_internalValue(_value) { QQmlEngine::setObjectOwnership(this, QQmlEngine::JavaScriptOwnership); }
	~QBigInt() {}

	/// @returns the current used big integer.
	BigIntVariant internalValue() const { return m_internalValue; }
	/// @returns a string representation of the big integer used. Invokable from QML.
	Q_INVOKABLE QString value() const;
	/// hex value.
	Q_INVOKABLE QString hexValue() const { return QString::fromStdString(dev::toHex(dev::u256(value().toStdString()))); }
	/// Set the value of the BigInteger used. Will use u256 type. Invokable from QML.
	Q_INVOKABLE void setValue(QString const& _value) { m_internalValue = dev::jsToU256(_value.toStdString()); }
	Q_INVOKABLE void setBigInt(QString const& _value) { m_internalValue = bigint(_value.toStdString()); }
	void setBigInt(u256 const& _value) { m_internalValue = _value; }
	/// Subtract by @a _value. Invokable from QML.
	Q_INVOKABLE QBigInt* subtract(QBigInt* const& _value) const;
	/// Add @a _value to the current big integer. Invokable from QML.
	Q_INVOKABLE QBigInt* add(QBigInt* const& _value) const;
	/// Multiply by @a _value. Invokable from QML.
	Q_INVOKABLE QBigInt* multiply(QBigInt* const& _value) const;
	/// divide by @a _value. Invokable from QML.
	Q_INVOKABLE QBigInt* divide(QBigInt* const& _value) const;
	/// check if the current value satisfy the given type
	Q_INVOKABLE QVariantMap checkAgainst(QString const& _type) const;

protected:
	 BigIntVariant m_internalValue;
};

}
}


