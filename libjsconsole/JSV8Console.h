//
// Created by Marek Kotewicz on 28/04/15.
//

#pragma once

#include "JSConsole.h"

namespace dev
{
namespace eth
{

class JSV8Engine;
class JSV8Printer;

class JSV8Console : public JSConsole
{
public:
	JSV8Console();
	void repl() const;

private:
	JSV8Engine m_engine;
	JSV8Printer m_printer;
};

}
}
