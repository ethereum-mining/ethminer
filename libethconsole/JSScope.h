//
// Created by Marek Kotewicz on 27/04/15.
//

#pragma once

namespace dev
{
namespace eth
{

class JSScope
{
public:
	JSScope()
	{ };

	virtual ~JSScope()
	{ };

	virtual const char* evaluate(const char* _cstr) const = 0;
};

}
}
