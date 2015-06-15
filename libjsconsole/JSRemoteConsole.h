//
// Created by Marek Kotewicz on 15/06/15.
//

#pragma once

#include <libjsengine/JSV8Engine.h>
#include <libjsengine/JSV8Printer.h>
#include "JSConsole.h"

namespace dev
{
namespace eth
{

class JSRemoteConsole: public JSConsole<JSV8Engine, JSV8Printer>
{

};

}
}
