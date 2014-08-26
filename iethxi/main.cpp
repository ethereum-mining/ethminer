#include <QtQml/QQmlApplicationEngine>
#include <QtWidgets/QApplication>

int main(int argc, char *argv[])
{
	QApplication a(argc, argv);
	QQmlApplicationEngine app(QUrl("qrc:/Simple.qml"));
	return a.exec();
}
