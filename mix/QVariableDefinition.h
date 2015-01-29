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

	/// Return the associated declaration of this variable definition.
	Q_INVOKABLE QVariableDeclaration* declaration() const { return m_dec; }
	/// Return the variable value.
	QString value() const { return m_value; }
	Q_INVOKABLE void setValue(QString _value) { m_value = _value; }
	Q_INVOKABLE void setDeclaration(QVariableDeclaration* _dec) { m_dec = _dec; }
	virtual bytes encodeValue() = 0;
	virtual void decodeValue(std::string const& _rawValue) = 0;

private:
	QString m_value;
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
	void decodeValue(std::string const& _rawValue) override;
	QBigInt* toBigInt() { return new QBigInt(value()); }
};

class QRealType: public QVariableDefinition
{
	Q_OBJECT

public:
	QRealType() {}
	QRealType(QVariableDeclaration* _def, QString _value): QVariableDefinition(_def, _value) {}
	dev::bytes encodeValue() override;
	void decodeValue(std::string const& _rawValue) override;
};

class QStringType: public QVariableDefinition
{
	Q_OBJECT

public:
	QStringType() {}
	QStringType(QVariableDeclaration* _def, QString _value): QVariableDefinition(_def, _value) {}
	dev::bytes encodeValue() override;
	void decodeValue(std::string const& _rawValue) override;
};

class QHashType: public QVariableDefinition
{
	Q_OBJECT

public:
	QHashType() {}
	QHashType(QVariableDeclaration* _def, QString _value): QVariableDefinition(_def, _value) {}
	dev::bytes encodeValue() override;
	void decodeValue(std::string const& _rawValue) override;
};

class QBoolType: public QVariableDefinition
{
	Q_OBJECT

public:
	QBoolType() {}
	QBoolType(QVariableDeclaration* _def, QString _value): QVariableDefinition(_def, _value) {}
	dev::bytes encodeValue() override;
	void decodeValue(std::string const& _rawValue) override;
	bool toBool() { return value() != "0"; }
};

}
}

Q_DECLARE_METATYPE(dev::mix::QIntType*)
Q_DECLARE_METATYPE(dev::mix::QStringType*)
Q_DECLARE_METATYPE(dev::mix::QHashType*)
Q_DECLARE_METATYPE(dev::mix::QBoolType*)
