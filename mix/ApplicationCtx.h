#ifndef APPLICATIONCONTEXT_H
#define APPLICATIONCONTEXT_H

#include <QQmlApplicationEngine>

class ApplicationCtx : public QObject
{
    Q_OBJECT

public:
    ApplicationCtx(QQmlApplicationEngine*);
    ~ApplicationCtx();
    QQmlApplicationEngine* appEngine();
    static ApplicationCtx* GetInstance();
    static void SetApplicationContext(QQmlApplicationEngine*);
private:
    static ApplicationCtx* m_instance;
    QQmlApplicationEngine* m_applicationEngine;
public slots:
    void QuitApplication();
};

#endif // APPLICATIONCTX_H
