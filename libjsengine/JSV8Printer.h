//
// Created by Marek Kotewicz on 28/04/15.
//

#pragma once

#include "JSV8Engine.h"


namespace dev
{
namespace eth
{

class JSV8Printer
{
public:
	JSV8Printer(JSV8Engine const& _engine);
	const char* print(v8::Handle<v8::Value> const& _value) const;
};

}
}
