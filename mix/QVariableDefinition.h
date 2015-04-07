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
/** @file QVariableDefinition.h
 * @author Yann yann@ethdev.com
 * @date 2014
 */

#pragma once

#include <QObject>
#include <libdevcore/Common.h>

namespace dev
{
namespace mix
{
class QVariableDeclaration;

class QVariableDefinition: public QObject
{
	Q_OBJECT

	Q_PROPERTY(QString value READ value CONSTANT)
	Q_PROPERTY(QVariableDeclaration* declaration READ declaration CONSTANT)

public:
	QVariableDefinition() {}
	QVariableDefinition(QVariableDeclaration* _def, QString _value): QObject(), m_value(_value), m_dec(_def) {}

	/// Return the associated declaration of this variable definition. Invokable from QML.
	Q_INVOKABLE QVariableDeclaration* declaration() const { return m_dec; }
	/// Return the variable value.
	QString value() const { return m_value; }
	/// Set a new value for this instance. Invokable from QML.
	Q_INVOKABLE void setValue(QString _value) { m_value = _value; }
	/// Set a new Declaration for this instance. Invokable from QML.
	Q_INVOKABLE void setDeclaration(QVariableDeclaration* _dec) { m_dec = _dec; }
	/// Encode the current value in order to be used as function parameter.
	virtual bytes encodeValue() = 0;
	/// Decode the return value @a _rawValue.
	virtual void decodeValue(dev::bytes const& _rawValue) = 0;
	/// returns String representation of the encoded value.
	Q_INVOKABLE QString encodeValueAsString();

protected:
	QString m_value;

private:
	QVariableDeclaration* m_dec;
};




}
}

