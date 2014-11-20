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
/** @file CodeEditorExtensionMan.cpp
 * @author Yann yann@ethdev.com
 * @date 2014
 * Ethereum IDE client.
 */

#include <QQuickItem>
#include <QGraphicsObject>
#include <QtDeclarative/QDeclarativeEngine>
#include <QQmlEngine>
#include <QtDeclarative/QDeclarativeView>
#include <QQmlComponent>
#include <QQuickTextDocument>
#include "CodeEditorExtensionMan.h"
#include "ConstantCompilation.h"
#include "features.h"
#include "ApplicationCtx.h"
#include <libevm/VM.h>
using namespace dev;

CodeEditorExtensionManager::CodeEditorExtensionManager()
{    
}

void CodeEditorExtensionManager::loadEditor(QQuickItem* _editor)
{
    if (!_editor)
        return;
    try{
        QVariant doc = _editor->property("textDocument");
        if (doc.canConvert<QQuickTextDocument*>()) {
            QQuickTextDocument* qqdoc = doc.value<QQuickTextDocument*>();
            if (qqdoc) {
                m_doc = qqdoc->textDocument();
            }
        }
    }
    catch (dev::Exception const& exception){
        qDebug() << "unable to load editor: ";
        qDebug() << exception.what();
    }
}

void CodeEditorExtensionManager::initExtensions()
{
    try{
        //only one for now
        ConstantCompilation* compil = new ConstantCompilation(m_doc);
        if (compil->tabUrl() != "")
            compil->addContentOn(m_tabView);
        compil->start();
    }
    catch (dev::Exception const& exception){
        qDebug() << "unable to load extensions: ";
        qDebug() << exception.what();
    }
}

void CodeEditorExtensionManager::setEditor(QQuickItem* _editor){
    this->loadEditor(_editor);
    this->initExtensions();
}

void CodeEditorExtensionManager::setTabView(QQuickItem* _tabView){
    m_tabView = _tabView;
}
