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
/** @file TransactionListModel.cpp
 * @author Arkadiy Paronyan arkadiy@ethdev.com
 * @date 2014
 * Ethereum IDE client.
 */

#include <QObject>
#include <QQmlEngine>
#include "TransactionListModel.h"

namespace dev
{
namespace mix
{
TransactionListModel::TransactionListModel(QObject* _parent) :
	QAbstractListModel(_parent)
{
	m_transactions.push_back(Transaction(0, "testTr", 0, 0, 0));
}

QHash<int, QByteArray> TransactionListModel::roleNames() const
{
	QHash<int, QByteArray> roles;
	roles[TitleRole] = "title";
	roles[IdRole] = "transactionId";
	return roles;
}

int TransactionListModel::rowCount(QModelIndex const& _parent) const
{
	Q_UNUSED(_parent);
	return m_transactions.size();
}

QVariant TransactionListModel::data(QModelIndex const& _index, int role) const
{
	if(_index.row() < 0 || _index.row() >= (int)m_transactions.size())
		return QVariant();
	auto const& transaction =  m_transactions.at(_index.row());
	switch(role)
	{
	case TitleRole:
		return QVariant(transaction.title);
	case IdRole:
		return QVariant(transaction.id);
	default:
		return QVariant();
	}
}

QObject* TransactionListModel::getItem(int _index)
{
	Transaction const& transaction = (_index >=0 && _index < (int)m_transactions.size()) ? m_transactions[_index] : Transaction();
	QObject* item = new TransactionListItem(transaction, nullptr);
	QQmlEngine::setObjectOwnership(item, QQmlEngine::JavaScriptOwnership);
	return item;
}

void TransactionListModel::edit(QObject* _data)
{
	//these properties come from TransactionDialog QML object
	int id = _data->property("transactionId").toInt();
	const QString title = _data->property("transactionTitle").toString();


	if (id >= 0 && id < (int)m_transactions.size())
	{
		beginRemoveRows(QModelIndex(), id, id);
		m_transactions.erase(m_transactions.begin() + id);
		endRemoveRows();
	}
	else
		id = rowCount(QModelIndex());

	beginInsertRows(QModelIndex(), id, id);
	m_transactions.push_back(Transaction(id, title, 0, 0, 0));
	emit transactionAdded();
	emit countChanged();
	endInsertRows();
}

int TransactionListModel::getCount() const
{
	return rowCount(QModelIndex());
}

}
}
