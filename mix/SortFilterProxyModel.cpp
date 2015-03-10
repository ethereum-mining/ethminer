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
/**
 * @author Yann <yann@ethdev.com>
 * @date 2015
 * Proxy used to filter a QML TableView.
 */


#include "SortFilterProxyModel.h"
#include <QtDebug>
#include <QtQml>

using namespace dev::mix;

SortFilterProxyModel::SortFilterProxyModel(QObject* _parent) : QSortFilterProxyModel(_parent)
{
	connect(this, &SortFilterProxyModel::rowsInserted, this, &SortFilterProxyModel::countChanged);
	connect(this, &SortFilterProxyModel::rowsRemoved, this, &SortFilterProxyModel::countChanged);
}

int SortFilterProxyModel::count() const
{
	return rowCount();
}

QObject* SortFilterProxyModel::source() const
{
	return sourceModel();
}

void SortFilterProxyModel::setSource(QObject* _source)
{
	setSourceModel(qobject_cast<QAbstractItemModel*>(_source));
}

QByteArray SortFilterProxyModel::sortRole() const
{
	return roleNames().value(QSortFilterProxyModel::sortRole());
}

void SortFilterProxyModel::setSortRole(QByteArray const& _role)
{
	QSortFilterProxyModel::setSortRole(roleKey(_role));
}

void SortFilterProxyModel::setSortOrder(Qt::SortOrder _order)
{
	QSortFilterProxyModel::sort(0, _order);
}

QString SortFilterProxyModel::filterString() const
{
	return filterRegExp().pattern();
}

void SortFilterProxyModel::setFilterString(QString const& _filter)
{
	setFilterRegExp(QRegExp(_filter, filterCaseSensitivity(), static_cast<QRegExp::PatternSyntax>(filterSyntax())));
}

SortFilterProxyModel::FilterSyntax SortFilterProxyModel::filterSyntax() const
{
	return static_cast<FilterSyntax>(filterRegExp().patternSyntax());
}

void SortFilterProxyModel::setFilterSyntax(SortFilterProxyModel::FilterSyntax _syntax)
{
	setFilterRegExp(QRegExp(filterString(), filterCaseSensitivity(), static_cast<QRegExp::PatternSyntax>(_syntax)));
}

QJSValue SortFilterProxyModel::get(int _idx) const
{
	QJSEngine *engine = qmlEngine(this);
	QJSValue value = engine->newObject();
	if (_idx >= 0 && _idx < count())
	{
		QHash<int, QByteArray> roles = roleNames();
		QHashIterator<int, QByteArray> it(roles);
		while (it.hasNext())
		{
			it.next();
			value.setProperty(QString::fromUtf8(it.value()), data(index(_idx, 0), it.key()).toString());
		}
	}
	return value;
}

int SortFilterProxyModel::roleKey(QByteArray const& _role) const
{
	QHash<int, QByteArray> roles = roleNames();
	QHashIterator<int, QByteArray> it(roles);
	while (it.hasNext())
	{
		it.next();
		if (it.value() == _role)
			return it.key();
	}
	return -1;
}

QHash<int, QByteArray> SortFilterProxyModel::roleNames() const
{
	if (QAbstractItemModel* source = sourceModel())
		return source->roleNames();
	return QHash<int, QByteArray>();
}

bool SortFilterProxyModel::filterAcceptsRow(int _sourceRow, QModelIndex const& _sourceParent) const
{
	QAbstractItemModel* model = sourceModel();
	QModelIndex sourceIndex = model->index(_sourceRow, 0, _sourceParent);
	if (!sourceIndex.isValid())
		return true;

	QString keyType = model->data(sourceIndex, roleKey(type.toUtf8())).toString();
	QString keyContent = model->data(sourceIndex, roleKey(content.toUtf8())).toString();
	return keyType.contains(m_filterType) && keyContent.contains(m_filterContent);
}

void SortFilterProxyModel::setFilterType(QString const& _type)
{
	m_filterType = QRegExp(_type, filterCaseSensitivity(), static_cast<QRegExp::PatternSyntax>(filterSyntax()));
	setFilterRegExp(_type);
}

QString SortFilterProxyModel::filterType() const
{
	return m_filterType.pattern();
}

void SortFilterProxyModel::setFilterContent(QString const& _content)
{
	m_filterContent = QRegExp(_content, filterCaseSensitivity(), static_cast<QRegExp::PatternSyntax>(filterSyntax()));
	setFilterRegExp(_content);
}

QString SortFilterProxyModel::filterContent() const
{
	return m_filterContent.pattern();
}

