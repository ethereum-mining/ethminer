#include "MainWin.h"
#include <QtWidgets/QApplication>

int main(int argc, char *argv[])
{
	QApplication a(argc, argv);
	Main w;
	w.show();
	
	return a.exec();
}
