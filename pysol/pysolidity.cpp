#include <Python.h>
#include "structmember.h"
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <vector>

#include "../libdevcore/CommonData.h"


#include <libsolidity/Compiler.h>
#include <libsolidity/CompilerStack.h>
#include <libsolidity/CompilerUtils.h>
#include <libsolidity/SourceReferenceFormatter.h>


std::string compile(std::string src) {
    dev::solidity::CompilerStack compiler;
    try
    {
        std::vector<uint8_t> m_data = compiler.compile(src, false);
        return std::string(m_data.begin(), m_data.end());
    }
    catch (dev::Exception const& exception)
    {
        std::ostringstream error;
        dev::solidity::SourceReferenceFormatter::printExceptionInformation(error, exception, "Error", compiler);
        std::string e = error.str();
        throw(e);
    }
}


std::string mk_full_signature(std::string src) {
    dev::solidity::CompilerStack compiler;
    try
    {
        compiler.compile(src);
        return compiler.getInterface("");
    }
    catch (dev::Exception const& exception)
    {
        std::ostringstream error;
        dev::solidity::SourceReferenceFormatter::printExceptionInformation(error, exception, "Error", compiler);
        std::string e = error.str();
        throw(e);
    }
}

std::string bob(std::string src) { return src + src; }

#define PYMETHOD(name, FROM, method, TO) \
    static PyObject * name(PyObject *, PyObject *args) { \
        try { \
        FROM(med) \
        return TO(method(med)); \
        } \
        catch (std::string e) { \
           PyErr_SetString(PyExc_Exception, e.c_str()); \
           return NULL; \
        } \
    }

#define FROMSTR(v) \
    const char *command; \
    int len; \
    if (!PyArg_ParseTuple(args, "s#", &command, &len)) \
        return NULL; \
    std::string v = std::string(command, len); \

// Convert string into python wrapper form
PyObject* pyifyString(std::string s) {
    return Py_BuildValue("s#", s.c_str(), s.length());
}

// Convert integer into python wrapper form
PyObject* pyifyInteger(unsigned int i) {
    return Py_BuildValue("i", i);
}

// Convert pyobject int into normal form
int cppifyInt(PyObject* o) {
    int out;
    if (!PyArg_Parse(o, "i", &out))
        throw("Argument should be integer");
    return out;
}

// Convert pyobject string into normal form
std::string cppifyString(PyObject* o) {
    const char *command;
    if (!PyArg_Parse(o, "s", &command))
        throw("Argument should be string");
    return std::string(command);
}

int fh(std::string i) {
    return dev::fromHex(i[0]);
}

PYMETHOD(ps_compile, FROMSTR, compile, pyifyString)
PYMETHOD(ps_mk_full_signature, FROMSTR, mk_full_signature, pyifyString)

static PyMethodDef PyextMethods[] = {
    {"compile",  ps_compile, METH_VARARGS,
        "Compile code."},
    {"mk_full_signature",  ps_mk_full_signature, METH_VARARGS,
        "Get the signature of a piece of code."},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

PyMODINIT_FUNC initsolidity(void)
{
     Py_InitModule( "solidity", PyextMethods );
}   
