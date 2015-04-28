//
// Created by Marek Kotewicz on 28/04/15.
//

#pragma once

#include "JSPrinter.h"
#include "JSV8Engine.h"


namespace dev
{
namespace eth
{

class JSV8Printer : public JSPrinter<JSV8Value>
{
public:
	JSV8Printer(JSV8Engine const& _engine);
	const char* prettyPrint(JSV8Value const& _value) const;
};

}
}
