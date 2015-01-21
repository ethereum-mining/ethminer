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
#include "StatusPane.h"

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

	Q_PROPERTY(QQuickItem* headerView MEMBER m_headerView WRITE setHeaderView)
	Q_PROPERTY(QQuickItem* rightView MEMBER m_rightView WRITE setRightView)

public:
	CodeEditorExtensionManager();
	~CodeEditorExtensionManager();
	/// Initialize all extensions.
	void initExtensions();
	/// Initialize extension.
	void initExtension(std::shared_ptr<Extension>);
	/// Set current tab view
	void setHeaderView(QQuickItem*);
	/// Set current right tab view.
	void setRightView(QQuickItem*);

private slots:
	void applyCodeHighlight();

private:
	QVector<std::shared_ptr<Extension>> m_features;
	QQuickItem* m_headerView;
	QQuickItem* m_rightView;
	AppContext* m_appContext;
	void loadEditor(QQuickItem* _editor);
};

}
}
