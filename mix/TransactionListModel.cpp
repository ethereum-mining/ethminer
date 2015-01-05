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
#include <QTextDocument>
#include <QAbstractListModel>
#include <libdevcore/CommonJS.h>
#include "TransactionListModel.h"
#include "QContractDefinition.h"
#include "QFunctionDefinition.h"
#include "QVariableDeclaration.h"


namespace dev
{
namespace mix
{

/// @todo Move this to QML
u256 fromQString(QString const& _s)
{
	return dev::jsToU256(_s.toStdString());
}

/// @todo Move this to QML
QString toQString(u256 _value)
{
	std::ostringstream s;
	s << _value;
	return QString::fromStdString(s.str());
}

TransactionListItem::TransactionListItem(int _index, TransactionSettings const& _t, QObject* _parent):
	QObject(_parent), m_index(_index), m_title(_t.title), m_functionId(_t.functionId), m_value(toQString(_t.value)),
	m_gas(toQString(_t.gas)), m_gasPrice(toQString(_t.gasPrice))
{}

TransactionListModel::TransactionListModel(QObject* _parent, QTextDocument* _document):
	QAbstractListModel(_parent), m_document(_document)
{
	qRegisterMetaType<TransactionListItem*>("TransactionListItem*");
}

QHash<int, QByteArray> TransactionListModel::roleNames() const
{
	QHash<int, QByteArray> roles;
	roles[TitleRole] = "title";
	roles[IdRole] = "transactionIndex";
	return roles;
}

int TransactionListModel::rowCount(QModelIndex const& _parent) const
{
	Q_UNUSED(_parent);
	return m_transactions.size();
}

QVariant TransactionListModel::data(QModelIndex const& _index, int _role) const
{
	if (_index.row() < 0 || _index.row() >= (int)m_transactions.size())
		return QVariant();
	auto const& transaction =  m_transactions.at(_index.row());
	switch (_role)
	{
	case TitleRole:
		return QVariant(transaction.title);
	case IdRole:
		return QVariant(_index.row());
	default:
		return QVariant();
	}
}

///@todo: get parameters from code model
QList<TransactionParameterItem*> buildParameters(QTextDocument* _document, TransactionSettings const& _transaction, QString const& _functionId)
{
	QList<TransactionParameterItem*> params;
	try
	{
		std::shared_ptr<QContractDefinition> contract = QContractDefinition::Contract(_document->toPlainText());
		auto functions = contract->functions();
		for (auto f : functions)
		{
			if (f->name() != _functionId)
				continue;

			auto parameters = f->parameters();
			//build a list of parameters for a function. If the function is selected as current, add parameter values as well
			for (auto p : parameters)
			{
				QString paramValue;
				if (f->name() == _transaction.functionId)
				{
					auto paramValueIter = _transaction.parameterValues.find(p->name());
					if (paramValueIter != _transaction.parameterValues.cend())
						paramValue = toQString(paramValueIter->second);
				}

				TransactionParameterItem* item = new TransactionParameterItem(p->name(), p->type(), paramValue);
				QQmlEngine::setObjectOwnership(item, QQmlEngine::JavaScriptOwnership);
				params.append(item);
			}
		}
	}
	catch (boost::exception const&)
	{
		//TODO:
	}

	return params;
}

///@todo: get fnctions from code model
QList<QString> TransactionListModel::getFunctions()
{
	QList<QString> functionNames;
	try
	{
		QString code = m_document->toPlainText();
		std::shared_ptr<QContractDefinition> contract(QContractDefinition::Contract(code));
		auto functions = contract->functions();
		for (auto f : functions)
		{
			functionNames.append(f->name());
		}
	}
	catch (boost::exception const&)
	{
	}
	return functionNames;
}

QVariantList TransactionListModel::getParameters(int _index, QString const& _functionId)
{
	TransactionSettings const& transaction = (_index >= 0 && _index < (int)m_transactions.size()) ? m_transactions[_index] : TransactionSettings();
	auto plist = buildParameters(m_document, transaction, _functionId);
	QVariantList vl;
	for (QObject* p : plist)
		vl.append(QVariant::fromValue(p));
	return vl;
}

TransactionListItem* TransactionListModel::getItem(int _index)
{
	TransactionSettings const& transaction = (_index >= 0 && _index < (int)m_transactions.size()) ? m_transactions[_index] : TransactionSettings();
	TransactionListItem* item = new TransactionListItem(_index, transaction, nullptr);
	QQmlEngine::setObjectOwnership(item, QQmlEngine::JavaScriptOwnership);
	return item;
}

void TransactionListModel::edit(QObject* _data)
{
	//these properties come from TransactionDialog QML object
	///@todo change the model to a qml component
	int index = _data->property("transactionIndex").toInt();
	QString title = _data->property("transactionTitle").toString();
	QString gas = _data->property("gas").toString();
	QString gasPrice = _data->property("gasPrice").toString();
	QString value = _data->property("transactionValue").toString();
	QString functionId = _data->property("functionId").toString();
	QAbstractListModel* paramsModel = qvariant_cast<QAbstractListModel*>(_data->property("transactionParams"));
	TransactionSettings transaction(title, functionId, fromQString(value), fromQString(gas), fromQString(gasPrice));
	int paramCount = paramsModel->rowCount(QModelIndex());
	for (int p = 0; p < paramCount; ++p)
	{
		QString paramName = paramsModel->data(paramsModel->index(p, 0), Qt::DisplayRole).toString();
		QString paramValue = paramsModel->data(paramsModel->index(p, 0), Qt::DisplayRole + 2).toString();
		if (!paramValue.isEmpty() && !paramName.isEmpty())
			transaction.parameterValues[paramName] = fromQString(paramValue);
	}

	if (index >= 0 && index < (int)m_transactions.size())
	{
		beginRemoveRows(QModelIndex(), index, index);
		m_transactions.erase(m_transactions.begin() + index);
		endRemoveRows();
	}
	else
		index = rowCount(QModelIndex());

	beginInsertRows(QModelIndex(), index, index);
	m_transactions.push_back(transaction);
	emit countChanged();
	endInsertRows();
}

int TransactionListModel::getCount() const
{
	return rowCount(QModelIndex());
}

void TransactionListModel::runTransaction(int _index)
{
	TransactionSettings tr = m_transactions.at(_index);
	emit transactionStarted(tr);
}


}
}

