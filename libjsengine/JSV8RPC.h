//
// Created by Marek Kotewicz on 04/05/15.
//

#pragma once

#include <libjsengine/JSV8Engine.h>

namespace dev
{
namespace eth
{

class JSV8RPC
{
public:
	JSV8RPC(JSV8Engine const& _engine);

	virtual void onSend(const char* _payload) = 0;
	const char* m_lastResponse;

private:
	JSV8Engine const& m_engine;
};

}
}
