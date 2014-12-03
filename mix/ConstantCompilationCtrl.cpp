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
/** @file ConstantCompilation.cpp
 * @author Yann yann@ethdev.com
 * @date 2014
 * Ethereum IDE client.
 */

#include <QQuickItem>
#include <QtCore/QFileInfo>
#include <QApplication>
#include <QQmlApplicationEngine>
#include <QtCore/QtCore>
#include <QDebug>
#include "ConstantCompilationCtrl.h"
#include "ConstantCompilationModel.h"
using namespace dev::mix;

ConstantCompilationCtrl::ConstantCompilationCtrl(QTextDocument* _doc)
{
	m_editor = _doc;
	m_compilationModel = new ConstantCompilationModel();
}

ConstantCompilationCtrl::~ConstantCompilationCtrl()
{
	delete m_compilationModel;
}

QString ConstantCompilationCtrl::contentUrl() const
{
	return QStringLiteral("qrc:/qml/BasicContent.qml");
}

QString ConstantCompilationCtrl::title() const
{
	return "compiler";
}

void ConstantCompilationCtrl::start() const
{
	connect(m_editor, SIGNAL(contentsChange(int,int,int)), this, SLOT(compile()));
}

void ConstantCompilationCtrl::compile()
{
	QString codeContent = m_editor->toPlainText().replace("\n", "");
	if (codeContent.isEmpty())
	{
		resetOutPut();
		return;
	}
	CompilerResult res = m_compilationModel->compile(m_editor->toPlainText());
	writeOutPut(res);
}

void ConstantCompilationCtrl::resetOutPut()
{
	QObject* status = m_view->findChild<QObject*>("status", Qt::FindChildrenRecursively);
	QObject* content = m_view->findChild<QObject*>("content", Qt::FindChildrenRecursively);
	status->setProperty("text", "");
	content->setProperty("text", "");
}

void ConstantCompilationCtrl::writeOutPut(CompilerResult const& _res)
{
	QObject* status = m_view->findChild<QObject*>("status", Qt::FindChildrenRecursively);
	QObject* content = m_view->findChild<QObject*>("content", Qt::FindChildrenRecursively);
	if (_res.success)
	{
		status->setProperty("text", "succeeded");
		status->setProperty("color", "green");
		content->setProperty("text", _res.hexCode);
		qDebug() << QString("compile succeeded " + _res.hexCode);
	}
	else
	{
		status->setProperty("text", "failure");
		status->setProperty("color", "red");
		content->setProperty("text", _res.comment);
		qDebug() << QString("compile failed " + _res.comment);
	}
}
