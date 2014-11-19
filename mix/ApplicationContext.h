#ifndef APPLICATIONCONTEXT_H
#define APPLICATIONCONTEXT_H

#include <QQmlApplicationEngine>

class ApplicationContext : public QObject
{
    Q_OBJECT

public:
    ApplicationContext(QQmlApplicationEngine*);
    ~ApplicationContext();
    QQmlApplicationEngine* appEngine();
    static ApplicationContext* GetInstance();
    static void SetApplicationContext(QQmlApplicationEngine*);
private:
    static ApplicationContext* m_instance;
    QQmlApplicationEngine* m_applicationEngine;
public slots:
    void QuitApplication();
};

#endif // APPLICATIONCONTEXT_H
