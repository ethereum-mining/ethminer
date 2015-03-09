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

#ifndef SORTFILTERPROXYMODEL_H
#define SORTFILTERPROXYMODEL_H

#include <QtCore/qsortfilterproxymodel.h>
#include <QtQml/qjsvalue.h>

namespace dev
{
namespace mix
{

class SortFilterProxyModel : public QSortFilterProxyModel
{
	Q_OBJECT
	Q_PROPERTY(int count READ count NOTIFY countChanged)
	Q_PROPERTY(QObject *source READ source WRITE setSource)

	Q_PROPERTY(QByteArray sortRole READ sortRole WRITE setSortRole)
	Q_PROPERTY(Qt::SortOrder sortOrder READ sortOrder WRITE setSortOrder)

	Q_PROPERTY(QString filterContent READ filterContent WRITE setFilterContent)
	Q_PROPERTY(QString filterType READ filterType WRITE setFilterType)
	Q_PROPERTY(QString filterString READ filterString WRITE setFilterString)
	Q_PROPERTY(FilterSyntax filterSyntax READ filterSyntax WRITE setFilterSyntax)

	Q_ENUMS(FilterSyntax)

public:
	explicit SortFilterProxyModel(QObject *parent = 0);

	QObject *source() const;
	void setSource(QObject *source);

	QByteArray sortRole() const;
	void setSortRole(const QByteArray &role);

	void setSortOrder(Qt::SortOrder order);

	QString filterContent() const;
	void setFilterContent(const QString &_content);
	QString filterType() const;
	void setFilterType(const QString &_type);

	/*QStringList filterRoles() const;
	void setFilterRoles(const QStringList &roles);
*/
	QString filterString() const;
	void setFilterString(const QString &filter);

	enum FilterSyntax {
		RegExp,
		Wildcard,
		FixedString
	};

	FilterSyntax filterSyntax() const;
	void setFilterSyntax(FilterSyntax syntax);

	int count() const;
	Q_INVOKABLE QJSValue get(int index) const;

signals:
	void countChanged();

protected:
	int roleKey(const QByteArray &role) const;
	QHash<int, QByteArray> roleNames() const;
	bool filterAcceptsRow(int sourceRow, const QModelIndex &sourceParent) const;

private:
	QRegExp m_filterType;
	QRegExp m_filterContent;
	const QString type = "type";
	const QString content = "content";
};

}
}
#endif // SORTFILTERPROXYMODEL_H
