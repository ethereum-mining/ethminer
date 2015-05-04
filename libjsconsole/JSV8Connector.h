//
// Created by Marek Kotewicz on 04/05/15.
//

#pragma once

#include <jsonrpccpp/server/abstractserverconnector.h>
#include <libjsengine/JSV8RPC.h>

namespace dev
{
namespace eth
{

class JSV8Connector : public jsonrpc::AbstractServerConnector, public JSV8RPC
{

public:
	JSV8Connector(JSV8Engine const &_engine) : JSV8RPC(_engine) {}
	virtual ~JSV8Connector();

	bool StartListening();
	bool StopListening();
	bool SendResponse(std::string const& _response, void* _addInfo = NULL);

	void onSend(const char* payload);
};

}
}
