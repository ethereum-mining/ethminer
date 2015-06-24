//
// Created by Marek Kotewicz on 15/06/15.
//

#pragma once

#include <string>
#include <libjsengine/JSV8RPC.h>
#include "CURLRequest.h"

namespace dev
{
namespace eth
{

class JSV8RemoteConnector : public JSV8RPC
{

public:
	JSV8RemoteConnector(JSV8Engine const& _engine, std::string _url): JSV8RPC(_engine), m_url(_url) {}
	virtual ~JSV8RemoteConnector() {}

	// implement JSV8RPC interface
	void onSend(char const* _payload);

private:
	std::string m_url;
	CURLRequest m_request;
};

}
}
