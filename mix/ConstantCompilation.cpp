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

#include "ConstantCompilation.h"
#include <QQuickItem>
#include <QtCore/QFileInfo>
#include <QApplication>
#include <QQmlApplicationEngine>
#include <QtCore/QtCore>
#include <QDebug>
#include <libevm/VM.h>
#include <libsolidity/Scanner.h>
#include <libsolidity/CompilerStack.h>
#include <libsolidity/SourceReferenceFormatter.h>
using namespace std;
using namespace dev;
using namespace dev::eth;

ConstantCompilation::ConstantCompilation(QTextDocument* _doc)
{
    m_editor = _doc;
}

QString ConstantCompilation::tabUrl(){
    return QStringLiteral("qrc:/qml/BasicContent.qml");
}

void ConstantCompilation::start()
{
    connect(m_editor, SIGNAL(contentsChange(int,int,int)), this, SLOT(compile()));
}

QString ConstantCompilation::title()
{
    return "compiler";
}

void ConstantCompilation::compile()
{
    QString codeContent = m_editor->toPlainText();
    if (codeContent == ""){
        this->writeOutPut(true, codeContent);
        return;
    }
    dev::solidity::CompilerStack compiler;
    dev::bytes m_data;
    QString content;
    try
    {
        m_data = compiler.compile(codeContent.toStdString(), true);
        content = QString::fromStdString(dev::eth::disassemble(m_data));
        this->writeOutPut(true, content);
    }
    catch (dev::Exception const& exception)
    {
        ostringstream error;
        solidity::SourceReferenceFormatter::printExceptionInformation(error, exception, "Error", compiler.getScanner());
        content = QString::fromStdString(error.str()).toHtmlEscaped();
        this->writeOutPut(false, content);
    }
    catch (...)
    {
        content = "Uncaught exception.";
        this->writeOutPut(false, content);
    }
}

void ConstantCompilation::writeOutPut(bool _success, QString _content){
    QObject* status = m_view->findChild<QObject*>("status", Qt::FindChildrenRecursively);
    QObject* content = m_view->findChild<QObject*>("content", Qt::FindChildrenRecursively);
    if (_content == ""){
        status->setProperty("text", "");
        content->setProperty("text", "");
    }
    else if (_success){
        status->setProperty("text", "succeeded");
        status->setProperty("color", "green");
        content->setProperty("text", _content);
        qDebug() << QString("compile succeeded " + _content);
    }
    else {
        status->setProperty("text", "failure");
        status->setProperty("color", "red");
        content->setProperty("text", _content);
        qDebug() << QString("compile failed " + _content);
    }
}




