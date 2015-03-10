/****************************************************************************
**
** Copyright (C) 2014 Digia Plc and/or its subsidiary(-ies).
** Contact: http://www.qt-project.org/legal
**
** This file is part of the examples of the Qt Toolkit.
**
** $QT_BEGIN_LICENSE:BSD$
** You may use this file under the terms of the BSD license as follows:
**
** "Redistribution and use in source and binary forms, with or without
** modification, are permitted provided that the following conditions are
** met:
**   * Redistributions of source code must retain the above copyright
**     notice, this list of conditions and the following disclaimer.
**   * Redistributions in binary form must reproduce the above copyright
**     notice, this list of conditions and the following disclaimer in
**     the documentation and/or other materials provided with the
**     distribution.
**   * Neither the name of Digia Plc and its Subsidiary(-ies) nor the names
**     of its contributors may be used to endorse or promote products derived
**     from this software without specific prior written permission.
**
**
** THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
** "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
** LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
** A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
** OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
** SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
** LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
** DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
** THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
** (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
** OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE."
**
** $QT_END_LICENSE$
**
****************************************************************************/

#include "sortfilterproxymodel.h"
#include <QtDebug>
#include <QtQml>

using namespace dev::mix;

SortFilterProxyModel::SortFilterProxyModel(QObject* _parent) : QSortFilterProxyModel(_parent)
{
	connect(this, SIGNAL(rowsInserted(QModelIndex,int,int)), this, SIGNAL(countChanged()));
	connect(this, SIGNAL(rowsRemoved(QModelIndex,int,int)), this, SIGNAL(countChanged()));
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
	setSourceModel(qobject_cast<QAbstractItemModel *>(_source));
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
	if (_idx >= 0 && _idx < count()) {
		QHash<int, QByteArray> roles = roleNames();
		QHashIterator<int, QByteArray> it(roles);
		while (it.hasNext()) {
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
	while (it.hasNext()) {
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

	QRegExp rx = filterRegExp();
	QAbstractItemModel *model = sourceModel();
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

