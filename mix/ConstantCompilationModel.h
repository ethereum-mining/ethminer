#ifndef CONSTANTCOMPILATIONMODEL_H
#define CONSTANTCOMPILATIONMODEL_H
#include <QObject>

struct compilerResult{
    QString hexCode;
    QString comment;
    bool success;
};

class ConstantCompilationModel
{

public:
    ConstantCompilationModel();
    compilerResult compile(QString code);
};

#endif // CONSTANTCOMPILATIONMODEL_H
