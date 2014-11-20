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
/** @file Feature.cpp
 * @author Yann yann@ethdev.com
 * @date 2014
 * Ethereum IDE client.
 */

#include "Feature.h"
#include "ApplicationCtx.h"
#include <libevm/VM.h>
#include <QMessageBox>
#include <QDebug>
using namespace dev;

Feature::Feature()
{
}

void Feature::addContentOn(QObject* tabView) {
    try{
        if (tabUrl() == "")
            return;

        QVariant returnValue;
        QQmlComponent* component = new QQmlComponent(
                    ApplicationCtx::GetInstance()->appEngine(),
                    QUrl(this->tabUrl()), tabView);

        QMetaObject::invokeMethod(tabView, "addTab",
                                Q_RETURN_ARG(QVariant, returnValue),
                                Q_ARG(QVariant, this->title()),
                                Q_ARG(QVariant, QVariant::fromValue(component)));

        m_view = qvariant_cast<QObject*>(returnValue);
    }
    catch (dev::Exception const& exception){
        qDebug() << exception.what();
    }
}

