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


#pragma once

#include <QtCore/qsortfilterproxymodel.h>
#include <QtQml/qjsvalue.h>

namespace dev
{
namespace mix
{

class SortFilterProxyModel: public QSortFilterProxyModel
{
	Q_OBJECT
	Q_PROPERTY(int count READ count NOTIFY countChanged)
	Q_PROPERTY(QObject* source READ source WRITE setSource)

	Q_PROPERTY(QByteArray sortRole READ sortRole WRITE setSortRole)
	Q_PROPERTY(Qt::SortOrder sortOrder READ sortOrder WRITE setSortOrder)

	Q_PROPERTY(QString filterContent READ filterContent WRITE setFilterContent)
	Q_PROPERTY(QString filterType READ filterType WRITE setFilterType)
	Q_PROPERTY(QString filterString READ filterString WRITE setFilterString)
	Q_PROPERTY(FilterSyntax filterSyntax READ filterSyntax WRITE setFilterSyntax)

	Q_ENUMS(FilterSyntax)

public:
	explicit SortFilterProxyModel(QObject* _parent = 0);

	QObject* source() const;
	void setSource(QObject* _source);

	QByteArray sortRole() const;
	void setSortRole(QByteArray const& _role);

	void setSortOrder(Qt::SortOrder _order);

	QString filterContent() const;
	void setFilterContent(QString const& _content);
	QString filterType() const;
	void setFilterType(QString const& _type);

	QString filterString() const;
	void setFilterString(QString const& _filter);

	enum FilterSyntax {
		RegExp,
		Wildcard,
		FixedString
	};

	FilterSyntax filterSyntax() const;
	void setFilterSyntax(FilterSyntax _syntax);

	int count() const;
	Q_INVOKABLE QJSValue get(int _index) const;

signals:
	void countChanged();

protected:
	int roleKey(QByteArray const& _role) const;
	QHash<int, QByteArray> roleNames() const;
	bool filterAcceptsRow(int _sourceRow, QModelIndex const& _sourceParent) const;

private:
	QRegExp m_filterType;
	QRegExp m_filterContent;
	const QString type = "type";
	const QString content = "content";
};

}
}
