#include "MainWin.h"
#include <QtWidgets/QApplication>

int main(int argc, char *argv[])
{
	QApplication a(argc, argv);
	Q_INIT_RESOURCE(js);
	Main w;
	w.show();
	
	return a.exec();
}
