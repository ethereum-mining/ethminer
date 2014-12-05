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
/** @file CodeEditorExtensionMan.h
 * @author Yann yann@ethdev.com
 * @date 2014
 * Ethereum IDE client.
 */

#pragma once

#include "memory"
#include <QQuickItem>
#include <QTextDocument>
#include <QVector>
#include "ConstantCompilationCtrl.h"

namespace dev
{

namespace mix
{

class CodeEditorExtensionManager: public QObject
{
	Q_OBJECT

	Q_PROPERTY(QQuickItem* editor MEMBER m_editor WRITE setEditor)
	Q_PROPERTY(QQuickItem* tabView MEMBER m_tabView WRITE setTabView)

public:
	CodeEditorExtensionManager() {}
	~CodeEditorExtensionManager();
	void initExtensions();
	void setEditor(QQuickItem*);
	void setTabView(QQuickItem*);

private:
	QQuickItem* m_editor;
	QVector<std::shared_ptr<ConstantCompilationCtrl>> m_features;
	QQuickItem* m_tabView;
	QTextDocument* m_doc;
	void loadEditor(QQuickItem*);
};

}

}
