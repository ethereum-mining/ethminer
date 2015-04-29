//
// Created by Marek Kotewicz on 28/04/15.
//

#include <string>
#include "JSV8Printer.h"
#include "libjsengine/JSEngineResources.hpp"

using namespace std;
using namespace dev;
using namespace eth;

JSV8Printer::JSV8Printer(JSV8Engine const& _engine)
{
	JSEngineResources resources;
	string prettyPrint = resources.loadResourceAsString("pretty_print");
	_engine.eval(prettyPrint.c_str());
}

const char* JSV8Printer::prettyPrint(JSV8Value const& _value) const
{
	return nullptr;
}
