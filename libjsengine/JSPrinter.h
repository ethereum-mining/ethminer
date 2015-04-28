//
// Created by Marek Kotewicz on 28/04/15.
//

#pragma once

namespace dev
{
namespace eth
{

template <typename T>
class JSPrinter
{
public:
	virtual const char* print(T const& _value) const { return _value.asCString(); }
	virtual const char* prettyPrint(T const& _value) const { return print(_value); }
};

}
}

