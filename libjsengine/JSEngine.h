//
// Created by Marek Kotewicz on 27/04/15.
//

#pragma once

namespace dev
{
namespace eth
{

class JSValue
{
public:
	virtual const char* asCString() const = 0;
};

template <typename T>
class JSEngine
{
public:
	// should be used to evalute javascript expression
	virtual T eval(const char* _cstr) const = 0;
};

}
}
