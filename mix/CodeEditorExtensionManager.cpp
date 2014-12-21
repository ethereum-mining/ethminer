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
/** @file CodeEditorExtensionManager.cpp
 * @author Yann yann@ethdev.com
 * @date 2014
 * Ethereum IDE client.
 */

#include <QQuickItem>
#include <QGraphicsObject>
#include <QQmlEngine>
#include <QQmlComponent>
#include <QQuickTextDocument>
#include <libevm/VM.h>
#include "ConstantCompilationControl.h"
#include "AssemblyDebuggerControl.h"
#include "TransactionListView.h"
#include "AppContext.h"
#include "CodeEditorExtensionManager.h"
using namespace dev::mix;

CodeEditorExtensionManager::~CodeEditorExtensionManager()
{
	m_features.clear();
}

void CodeEditorExtensionManager::loadEditor(QQuickItem* _editor)
{
	if (!_editor)
		return;
	try
	{
		QVariant doc = _editor->property("textDocument");
		if (doc.canConvert<QQuickTextDocument*>())
		{
			QQuickTextDocument* qqdoc = doc.value<QQuickTextDocument*>();
			if (qqdoc)
			{
				m_doc = qqdoc->textDocument();
				auto args = QApplication::arguments();
				if (args.length() > 1)
				{
					QString path = args[1];
					QFile file(path);
					if (file.exists() && file.open(QFile::ReadOnly))
						m_doc->setPlainText(file.readAll());
				}
			}
		}
	}
	catch (...)
	{
		qDebug() << "unable to load editor: ";
	}
}

void CodeEditorExtensionManager::initExtensions()
{
	initExtension(std::make_shared<ConstantCompilationControl>(m_doc));
	std::shared_ptr<AssemblyDebuggerControl> debug = std::make_shared<AssemblyDebuggerControl>(m_doc);
	std::shared_ptr<TransactionListView> tr = std::make_shared<TransactionListView>(m_doc);
	QObject::connect(tr->model(), &TransactionListModel::transactionStarted, debug.get(), &AssemblyDebuggerControl::runTransaction);
	initExtension(debug);
	initExtension(tr);
}

void CodeEditorExtensionManager::initExtension(std::shared_ptr<Extension> _ext)
{
	if (!_ext->contentUrl().isEmpty())
	{
		try
		{
			if (_ext->getDisplayBehavior() == ExtensionDisplayBehavior::Tab)
				_ext->addTabOn(m_tabView);
			else if (_ext->getDisplayBehavior() == ExtensionDisplayBehavior::RightTab)
				_ext->addTabOn(m_rightTabView);
		}
		catch (...)
		{
			qDebug() << "Exception when adding tab into view.";
			return;
		}
	}
	_ext->start();
	m_features.append(_ext);
}

void CodeEditorExtensionManager::setEditor(QQuickItem* _editor)
{
	this->loadEditor(_editor);
	this->initExtensions();
}

void CodeEditorExtensionManager::setRightTabView(QQuickItem* _tabView)
{
	m_rightTabView = _tabView;
}

void CodeEditorExtensionManager::setTabView(QQuickItem* _tabView)
{
	m_tabView = _tabView;
}
