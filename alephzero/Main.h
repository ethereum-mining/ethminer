#ifndef MAIN_H
#define MAIN_H

#include <QDialog>

namespace Ui {
class Main;
}

class Main : public QDialog
{
	Q_OBJECT
	
public:
	explicit Main(QWidget *parent = 0);
	~Main();
	
private slots:
	void on_connect_clicked();

private:
	Client c;

	Ui::Main *ui;
};

#endif // MAIN_H
