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

namespace dev
{
namespace mix
{

struct TransacionParameterValue
{
	QVariant value;
};

struct Transaction
{
	Transaction():
		id(-1), value(0), gas(10000), gasPrice(10) {}

	Transaction(int _id, QString const& _title, u256 _value, u256 _gas, u256 _gasPrice):
		id(_id), title(_title), value(_value), gas(_gas), gasPrice(_gasPrice) {}

	int id;
	QString title;
	u256 value;
	u256 gas;
	u256 gasPrice;
	QString functionId;
	std::vector<TransacionParameterValue> parameterValues;
};

class TransactionListItem: public QObject
{
	Q_OBJECT
	Q_PROPERTY(int transactionId READ transactionId CONSTANT)
	Q_PROPERTY(QString title READ title CONSTANT)
	Q_PROPERTY(bool selected READ selected CONSTANT)

public:
	TransactionListItem(Transaction const& _t, QObject* _parent):
		QObject(_parent), m_id(_t.id), m_title(_t.title), m_selected(false) {}
	QString title() { return m_title; }
	int transactionId() { return m_id; }
	bool selected() { return m_selected; }

private:
	int m_id;
	QString m_title;
	bool m_selected;
};


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
	TransactionListModel(QObject* _parent);
	~TransactionListModel() {}

	QHash<int, QByteArray> roleNames() const override;
	int rowCount(QModelIndex const& _parent) const override;
	QVariant data(QModelIndex const& _index, int _role) const override;
	int getCount() const;
	Q_INVOKABLE void edit(QObject* _data);
	Q_INVOKABLE QObject* getItem(int _index);

signals:
	void transactionAdded();
	void countChanged();

private:
	std::vector<Transaction> m_transactions;
};

}

}
