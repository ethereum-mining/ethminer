//
// Created by Marek Kotewicz on 28/04/15.
//

#pragma once

#include "JSEngine.h"

namespace dev
{
namespace eth
{

class JSPrinter
{
public:
	virtual const char* print(JSValue const& _value) const { return _value.asCString(); }
};

}
}

