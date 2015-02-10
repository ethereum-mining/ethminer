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

#include <QAbstractListModel>
#include "QBigInt.h"
#include "QVariableDeclaration.h"

namespace dev
{
namespace mix
{

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

protected:
	QString m_value;

private:
	QVariableDeclaration* m_dec;
};

class QVariableDefinitionList: public QAbstractListModel
{
	Q_OBJECT

public:
	QVariableDefinitionList(QList<QVariableDefinition*> _def): m_def(_def) {}
	int rowCount(const QModelIndex& parent = QModelIndex()) const override;
	QVariant data(const QModelIndex& index, int role = Qt::DisplayRole) const override;
	QHash<int, QByteArray> roleNames() const override;
	/// Return the variable definition at index _idx.
	QVariableDefinition* val(int _idx);
	/// Return the list of variables.
	QList<QVariableDefinition*> def() { return m_def; }

private:
	QList<QVariableDefinition*> m_def;
};

class QIntType: public QVariableDefinition
{
	Q_OBJECT

public:
	QIntType() {}
	QIntType(QVariableDeclaration* _def, QString _value): QVariableDefinition(_def, _value) {}
	dev::bytes encodeValue() override;
	void decodeValue(dev::bytes const& _rawValue) override;
	/// @returns an instance of QBigInt for the current value.
	QBigInt* toBigInt() { return new QBigInt(m_bigIntvalue); }
	dev::bigint bigInt() { return m_bigIntvalue; }
	void setValue(dev::bigint _value);

private:
	dev::bigint m_bigIntvalue;
};

class QRealType: public QVariableDefinition
{
	Q_OBJECT

public:
	QRealType() {}
	QRealType(QVariableDeclaration* _def, QString _value): QVariableDefinition(_def, _value) {}
	dev::bytes encodeValue() override;
	void decodeValue(dev::bytes const& _rawValue) override;
};

class QStringType: public QVariableDefinition
{
	Q_OBJECT

public:
	QStringType() {}
	QStringType(QVariableDeclaration* _def, QString _value): QVariableDefinition(_def, _value) {}
	dev::bytes encodeValue() override;
	void decodeValue(dev::bytes const& _rawValue) override;
};

class QHashType: public QVariableDefinition
{
	Q_OBJECT

public:
	QHashType() {}
	QHashType(QVariableDeclaration* _def, QString _value): QVariableDefinition(_def, _value) {}
	dev::bytes encodeValue() override;
	void decodeValue(dev::bytes const& _rawValue) override;
};

class QBoolType: public QVariableDefinition
{
	Q_OBJECT

public:
	QBoolType() {}
	QBoolType(QVariableDeclaration* _def, QString _value): QVariableDefinition(_def, _value) {}
	dev::bytes encodeValue() override;
	void decodeValue(dev::bytes const& _rawValue) override;
	///  @returns the boolean value for the current definition.
	bool toBool() { return m_boolValue; }

private:
	bool m_boolValue;
};

}
}

Q_DECLARE_METATYPE(dev::mix::QIntType*)
Q_DECLARE_METATYPE(dev::mix::QStringType*)
Q_DECLARE_METATYPE(dev::mix::QHashType*)
Q_DECLARE_METATYPE(dev::mix::QBoolType*)
