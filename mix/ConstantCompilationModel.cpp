#include "ConstantCompilationModel.h"
#include <QObject>
#include <libevm/VM.h>
#include <libsolidity/Scanner.h>
#include <libsolidity/CompilerStack.h>
#include <libsolidity/SourceReferenceFormatter.h>
using namespace std;
using namespace dev;
using namespace dev::eth;

ConstantCompilationModel::ConstantCompilationModel()
{
}

compilerResult ConstantCompilationModel::compile(QString code)
{
    dev::solidity::CompilerStack compiler;
    dev::bytes m_data;
    compilerResult res;
    try
    {
        m_data = compiler.compile(code.toStdString(), true);
        res.success = true;
        res.comment = "ok";
        res.hexCode = QString::fromStdString(dev::eth::disassemble(m_data));

    }
    catch (dev::Exception const& exception)
    {
        ostringstream error;
        solidity::SourceReferenceFormatter::printExceptionInformation(error, exception, "Error", compiler.getScanner());
        res.success = false;
        res.comment = QString::fromStdString(error.str()).toHtmlEscaped();
        res.hexCode = "";
    }
    catch (...)
    {
        res.success = false;
        res.comment = "Uncaught exception.";
        res.hexCode = "";
    }
    return res;
}

