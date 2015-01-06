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
#include "ConstantCompilationControl.h"
#include "AssemblyDebuggerControl.h"
#include "StateListView.h"
#include "AppContext.h"
#include "MixApplication.h"
#include "CodeModel.h"
#include "CodeHighlighter.h"
#include "CodeEditorExtensionManager.h"

using namespace dev::mix;

CodeEditorExtensionManager::CodeEditorExtensionManager():
	m_appContext(static_cast<MixApplication*>(QApplication::instance())->context())
{
}

CodeEditorExtensionManager::~CodeEditorExtensionManager()
{
	m_features.clear();
}

void CodeEditorExtensionManager::loadEditor(QQuickItem* _editor)
{
	if (!_editor)
		return;

	QVariant doc = _editor->property("textDocument");
	if (doc.canConvert<QQuickTextDocument*>())
	{
		QQuickTextDocument* qqdoc = doc.value<QQuickTextDocument*>();
		if (qqdoc)
		{
			m_doc = qqdoc->textDocument();
		}
	}
}

void CodeEditorExtensionManager::initExtensions()
{
	std::shared_ptr<ConstantCompilationControl> output = std::make_shared<ConstantCompilationControl>(m_appContext);
	std::shared_ptr<AssemblyDebuggerControl> debug = std::make_shared<AssemblyDebuggerControl>(m_appContext);
	std::shared_ptr<StateListView> stateList = std::make_shared<StateListView>(m_appContext);
	QObject::connect(m_doc, &QTextDocument::contentsChange, this, &CodeEditorExtensionManager::onCodeChange);
	QObject::connect(debug.get(), &AssemblyDebuggerControl::runFailed, output.get(), &ConstantCompilationControl::displayError);
	QObject::connect(m_appContext->codeModel(), &CodeModel::compilationComplete, this, &CodeEditorExtensionManager::applyCodeHighlight);

	initExtension(output);
	initExtension(debug);
	initExtension(stateList);
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

	auto args = QApplication::arguments();
	if (args.length() > 1)
	{
		QString path = args[1];
		QFile file(path);
		if (file.exists() && file.open(QFile::ReadOnly))
			m_doc->setPlainText(file.readAll());
	}
}

void CodeEditorExtensionManager::onCodeChange()
{
	m_appContext->codeModel()->updateFormatting(m_doc); //update old formatting
	m_appContext->codeModel()->registerCodeChange(m_doc->toPlainText());
}

void CodeEditorExtensionManager::applyCodeHighlight()
{
	m_appContext->codeModel()->updateFormatting(m_doc);
}

void CodeEditorExtensionManager::setRightTabView(QQuickItem* _tabView)
{
	m_rightTabView = _tabView;
}

void CodeEditorExtensionManager::setTabView(QQuickItem* _tabView)
{
	m_tabView = _tabView;
}
