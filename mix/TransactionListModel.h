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
/** @file TransactionListView.h
 * @author Arkadiy Paronyan arkadiy@ethdev.com
 * @date 2014
 * Ethereum IDE client.
 */

#pragma once

#include <QObject>
#include <QVariant>
#include <QAbstractListModel>
#include <QHash>
#include <QByteArray>
#include <libdevcore/Common.h>
#include <libethcore/CommonEth.h>

class QTextDocument;

namespace dev
{
namespace mix
{

/// Backend transaction config class
struct TransactionSettings
{
	TransactionSettings():
		value(0), gas(10000), gasPrice(10 * dev::eth::szabo) {}

	TransactionSettings(QString const& _title, QString const& _functionId, u256 _value, u256 _gas, u256 _gasPrice):
		title(_title), functionId(_functionId), value(_value), gas(_gas), gasPrice(_gasPrice) {}

	/// User specified transaction title
	QString title;
	/// Contract function name
	QString functionId;
	/// Transaction value
	u256 value;
	/// Gas
	u256 gas;
	/// Gas price
	u256 gasPrice;
	/// Mapping from contract function parameter name to value
	std::map<QString, u256> parameterValues;
};

/// QML transaction parameter class
class TransactionParameterItem: public QObject
{
	Q_OBJECT
	Q_PROPERTY(QString name READ name CONSTANT)
	Q_PROPERTY(QString type READ type CONSTANT)
	Q_PROPERTY(QString value READ value CONSTANT)
public:
	TransactionParameterItem(QString const& _name, QString const& _type, QString const& _value):
		m_name(_name), m_type(_type), m_value(_value) {}

	/// Parameter name, set by contract definition
	QString name() { return m_name; }
	/// Parameter type, set by contract definition
	QString type() { return m_type; }
	/// Parameter value, set by user
	QString value() { return m_value; }

private:
	QString m_name;
	QString m_type;
	QString m_value;
};

class TransactionListItem: public QObject
{
	Q_OBJECT
	Q_PROPERTY(int index READ index CONSTANT)
	Q_PROPERTY(QString title READ title CONSTANT)
	Q_PROPERTY(QString functionId READ functionId CONSTANT)
	Q_PROPERTY(QString gas READ gas CONSTANT)
	Q_PROPERTY(QString gasPrice READ gasPrice CONSTANT)
	Q_PROPERTY(QString value READ value CONSTANT)

public:
	TransactionListItem(int _index, TransactionSettings const& _t, QObject* _parent);

	/// User specified transaction title
	QString title() { return m_title; }
	/// Gas
	QString gas() { return m_gas; }
	/// Gas cost
	QString gasPrice() { return m_gasPrice; }
	/// Transaction value
	QString value() { return m_value; }
	/// Contract function name
	QString functionId() { return m_functionId; }
	/// Index of this transaction in the transactions list
	int index() { return m_index; }

private:
	int m_index;
	QString m_title;
	QString m_functionId;
	QString m_value;
	QString m_gas;
	QString m_gasPrice;
};

/// QML model for a list of transactions
class TransactionListModel: public QAbstractListModel
{
	Q_OBJECT
	Q_PROPERTY(int count READ getCount() NOTIFY countChanged())

	enum Roles
	{
		TitleRole = Qt::DisplayRole,
		IdRole = Qt::UserRole + 1
	};

public:
	TransactionListModel(QObject* _parent, QTextDocument* _document);
	~TransactionListModel() {}

	QHash<int, QByteArray> roleNames() const override;
	int rowCount(QModelIndex const& _parent) const override;
	QVariant data(QModelIndex const& _index, int _role) const override;
	int getCount() const;
	/// Apply changes from transaction dialog. Argument is a dialog model as defined in TransactionDialog.qml
	/// @todo Change that to transaction item
	Q_INVOKABLE void edit(QObject* _data);
	/// @returns transaction item for a give index
	Q_INVOKABLE TransactionListItem* getItem(int _index);
	/// @returns a list of functions for current contract
	Q_INVOKABLE QList<QString> getFunctions();
	/// @returns function parameters along with parameter values if set. @see TransactionParameterItem
	Q_INVOKABLE QVariantList getParameters(int _id, QString const& _functionId);
	/// Launch transaction execution UI handler
	Q_INVOKABLE void runTransaction(int _index);

signals:
	/// Transaction count has changed
	void countChanged();
	/// Transaction has been launched
	void transactionStarted(dev::mix::TransactionSettings);

private:
	std::vector<TransactionSettings> m_transactions;
	QTextDocument* m_document;
};

}

}

