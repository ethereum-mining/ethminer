//
// Created by Marek Kotewicz on 28/04/15.
//

#pragma once

#include <libjsengine/JSV8Engine.h>
#include <libjsengine/JSV8Printer.h>

namespace dev
{
namespace eth
{

class JSConsole
{
public:
	JSConsole();
	void repl() const;

private:
	std::string promptForIndentionLevel(int _i) const;

	JSV8Engine m_engine;
	JSV8Printer m_printer;
};

}
}
