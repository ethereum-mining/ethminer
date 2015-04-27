//
// Created by Marek Kotewicz on 27/04/15.
//

#pragma once

namespace dev
{
namespace eth
{

class JSEngine
{
public:
	JSEngine() {};
	virtual ~JSEngine() {};
	// should be used to evalute javascript expression
	virtual const char* evaluate(const char* _cstr) const = 0;
};

}
}
