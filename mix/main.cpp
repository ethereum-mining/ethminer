#include <QApplication>
#include <QQmlApplicationEngine>
#include <QQuickItem>
#include "CodeEditorExtensionMan.h"
#include "ApplicationContext.h"

int main(int argc, char *argv[])
{
    QApplication app(argc, argv);
    QQmlApplicationEngine* engine = new QQmlApplicationEngine();
    qmlRegisterType<CodeEditorExtensionManager>("CodeEditorExtensionManager", 1, 0, "CodeEditorExtensionManager");

    ApplicationContext::SetApplicationContext(engine);
    QObject::connect(&app, SIGNAL(lastWindowClosed()), ApplicationContext::GetInstance(), SLOT(QuitApplication())); //use to kill ApplicationContext and other stuff

    engine->load(QUrl(QStringLiteral("qrc:/main.qml")));
    return app.exec();
}

