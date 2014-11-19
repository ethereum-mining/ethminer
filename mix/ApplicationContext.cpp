#include "ApplicationContext.h"
#include <QQmlApplicationEngine>

ApplicationContext* ApplicationContext::m_instance = nullptr;

ApplicationContext::ApplicationContext(QQmlApplicationEngine* _engine)
{
    m_applicationEngine = _engine;
}

ApplicationContext::~ApplicationContext()
{
    delete m_applicationEngine;
}

ApplicationContext* ApplicationContext::GetInstance()
{
    return m_instance;
}

void ApplicationContext::SetApplicationContext(QQmlApplicationEngine* engine)
{
    m_instance = new ApplicationContext(engine);
}

QQmlApplicationEngine* ApplicationContext::appEngine(){
    return m_applicationEngine;
}

void ApplicationContext::QuitApplication()
{
    delete m_instance;
}
