#include "Main.h"
#include "ui_Main.h"

Main::Main(QWidget *parent) :
	QDialog(parent),
	ui(new Ui::Main)
{
	setWindowFlags(Qt::Window);
	ui->setupUi(this);

	ui->transactions->setHtml("Hello world!");
}

Main::~Main()
{
	delete ui;
}
