//
// Created by Marek Kotewicz on 15/06/15.
//

#include "JSV8RemoteConnector.h"
#include "CURLRequest.h"

using namespace std;
using namespace dev;
using namespace dev::eth;

void JSV8RemoteConnector::onSend(char const* _payload)
{
	CURLRequest request;
	request.setUrl(m_url);
	request.setBody(_payload);
	long code;
	string response;
	tie(code, response) = request.post();
	(void)code;
	m_lastResponse = response.c_str();
}
