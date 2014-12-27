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
/** @file ConstantCompilationControl.cpp
 * @author Yann yann@ethdev.com
 * @date 2014
 * Ethereum IDE client.
 */

#include <QQmlContext>
#include <QQuickItem>
#include <QtCore/QFileInfo>
#include <QApplication>
#include <QQmlApplicationEngine>
#include <QtCore/QtCore>
#include <QDebug>
#include "ConstantCompilationControl.h"
#include "QContractDefinition.h"
#include "AppContext.h"
#include "CodeModel.h"

using namespace dev::mix;

ConstantCompilationControl::ConstantCompilationControl(AppContext* _context): Extension(_context, ExtensionDisplayBehavior::Tab)
{
	connect(_context->codeModel(), &CodeModel::compilationComplete, this, &ConstantCompilationControl::update);
	connect(_context->codeModel(), &CodeModel::compilationComplete, this, &ConstantCompilationControl::update);
}

QString ConstantCompilationControl::contentUrl() const
{
	return QStringLiteral("qrc:/qml/BasicContent.qml");
}

QString ConstantCompilationControl::title() const
{
	return QApplication::tr("compiler");
}

void ConstantCompilationControl::start() const
{
}

void ConstantCompilationControl::update()
{
	auto result = m_ctx->codeModel()->code();

	QObject* status = m_view->findChild<QObject*>("status", Qt::FindChildrenRecursively);
	QObject* content = m_view->findChild<QObject*>("content", Qt::FindChildrenRecursively);
	if (result->successfull())
	{
		status->setProperty("text", "succeeded");
		status->setProperty("color", "green");
		content->setProperty("text", result->assemblyCode());
	}
	else
	{
		status->setProperty("text", "failure");
		status->setProperty("color", "red");
		content->setProperty("text", result->compilerMessage());
	}
}

void ConstantCompilationControl::resetOutPut()
{
	QObject* status = m_view->findChild<QObject*>("status", Qt::FindChildrenRecursively);
	QObject* content = m_view->findChild<QObject*>("content", Qt::FindChildrenRecursively);
	status->setProperty("text", "");
	content->setProperty("text", "");
}


void ConstantCompilationControl::displayError(QString const& _error)
{
	QObject* status = m_view->findChild<QObject*>("status", Qt::FindChildrenRecursively);
	QObject* content = m_view->findChild<QObject*>("content", Qt::FindChildrenRecursively);
	status->setProperty("text", "failure");
	status->setProperty("color", "red");
	content->setProperty("text", _error);
}
