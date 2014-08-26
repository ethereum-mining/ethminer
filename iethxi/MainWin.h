#ifndef MAIN_H
#define MAIN_H

#include <QtQml/QQmlApplicationEngine>

class Main: public QObject
{
	Q_OBJECT
	
public:
	explicit Main(QWidget *parent = 0);
	~Main();

private:
	QQmlApplicationEngine* m_view;
};

#endif // MAIN_H
