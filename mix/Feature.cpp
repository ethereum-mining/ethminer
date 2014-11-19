#include "Feature.h"
#include "ApplicationContext.h"
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
                    ApplicationContext::GetInstance()->appEngine(),
                    QUrl(this->tabUrl()));

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

