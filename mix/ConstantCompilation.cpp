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
    return QStringLiteral("qrc:/BasicContent.qml");
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




