#ifndef CONSTANTCOMPILATION_H
#define CONSTANTCOMPILATION_H

#include <QTextDocument>
#include "Feature.h"

class ConstantCompilation : public Feature
{
    Q_OBJECT

public:
    ConstantCompilation(QTextDocument* doc);
    void start();
    QString title();
    QString tabUrl();

private:
    QTextDocument* m_editor;
    void writeOutPut(bool success, QString content);

public Q_SLOTS:
    void compile();

};

#endif // CONSTANTCOMPILATION_H
