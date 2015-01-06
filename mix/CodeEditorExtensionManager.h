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
/** @file CodeEditorExtensionManager.h
 * @author Yann yann@ethdev.com
 * @date 2014
 * Ethereum IDE client.
 */

#pragma once

#include <memory>
#include <QQuickItem>
#include <QTextDocument>
#include <QVector>
#include "ConstantCompilationControl.h"

namespace dev
{
namespace mix
{


class AppContext;

/**
 * @brief Init and provides connection between extensions.
 */
class CodeEditorExtensionManager: public QObject
{
	Q_OBJECT

	Q_PROPERTY(QQuickItem* editor MEMBER m_editor WRITE setEditor)
	Q_PROPERTY(QQuickItem* tabView MEMBER m_tabView WRITE setTabView)
	Q_PROPERTY(QQuickItem* rightTabView MEMBER m_rightTabView WRITE setRightTabView)

public:
	CodeEditorExtensionManager();
	~CodeEditorExtensionManager();
	/// Initialize all extensions.
	void initExtensions();
	/// Initialize extension.
	void initExtension(std::shared_ptr<Extension>);
	/// Set current text editor.
	void setEditor(QQuickItem*);
	/// Set current tab view
	void setTabView(QQuickItem*);
	/// Set current right tab view.
	void setRightTabView(QQuickItem*);

private slots:
	void onCodeChange();
	void applyCodeHighlight();

private:
	QQuickItem* m_editor;
	QVector<std::shared_ptr<Extension>> m_features;
	QQuickItem* m_tabView;
	QQuickItem* m_rightTabView;
	QTextDocument* m_doc;
	AppContext* m_appContext;
	void loadEditor(QQuickItem* _editor);
};

}
}
