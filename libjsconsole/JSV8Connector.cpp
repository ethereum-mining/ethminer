//
// Created by Marek Kotewicz on 04/05/15.
//

#include "JSV8Connector.h"

using namespace std;
using namespace dev;
using namespace dev::eth;

bool JSV8Connector::StartListening()
{
	return true;
}

bool JSV8Connector::StopListening()
{
	return true;
}

bool JSV8Connector::SendResponse(std::string const &_response, void *_addInfo)
{
	(void)_addInfo;
	m_lastResponse = _response.c_str();
	return true;
}

void JSV8Connector::onSend(const char *payload)
{
	OnRequest(payload, NULL);
}

JSV8Connector::~JSV8Connector()
{
	StopListening();
}
