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
#include "StatusPane.h"
#include "AppContext.h"
#include "MixApplication.h"
#include "CodeModel.h"
#include "ClientModel.h"
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
}

void CodeEditorExtensionManager::initExtensions()
{
	std::shared_ptr<StatusPane> output = std::make_shared<StatusPane>(m_appContext);
	QObject::connect(m_appContext->codeModel(), &CodeModel::compilationComplete, this, &CodeEditorExtensionManager::applyCodeHighlight);

	initExtension(output);
}

void CodeEditorExtensionManager::initExtension(std::shared_ptr<Extension> _ext)
{
	if (!_ext->contentUrl().isEmpty())
	{
		try
		{
			if (_ext->getDisplayBehavior() == ExtensionDisplayBehavior::RightView)
				_ext->addTabOn(m_rightView);
			if (_ext->getDisplayBehavior() == ExtensionDisplayBehavior::HeaderView)
				_ext->addTabOn(m_headerView);
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

void CodeEditorExtensionManager::applyCodeHighlight()
{
	//TODO: reimplement
}

void CodeEditorExtensionManager::setRightView(QQuickItem* _rightView)
{
	m_rightView = _rightView;
}

void CodeEditorExtensionManager::setHeaderView(QQuickItem* _headerView)
{
	m_headerView = _headerView;
	initExtensions(); //TODO: move this to a proper place
}
