//
// Created by Marek Kotewicz on 04/05/15.
//

#pragma once

//#include <jsonrpccpp/server/abstractserverconnector.h>
#include <libjsengine/JSV8Engine.h>

namespace dev
{
namespace eth
{

class JSV8RPC
{
public:
	JSV8RPC(JSV8Engine const& _engine);
	virtual ~JSV8RPC();

	bool StartListening();

	bool StopListening();

	bool SendResponse(std::string const& _response, void* _addInfo = NULL);

private:
	JSV8Engine const& m_engine;
	std::string m_lastResponse;
};

}
}
