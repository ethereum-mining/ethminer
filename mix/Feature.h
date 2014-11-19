#ifndef FEATURE_H
#define FEATURE_H

#include <QApplication>
#include <QQmlComponent>

class Feature : public QObject
{
    Q_OBJECT

public:
    Feature();
    virtual QString tabUrl() { return ""; }
    virtual QString title() { return ""; }
    void addContentOn(QObject* tabView);

protected:
    QObject* m_view;
};

#endif // FEATURE_H
