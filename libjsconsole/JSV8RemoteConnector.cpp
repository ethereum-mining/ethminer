//
// Created by Marek Kotewicz on 15/06/15.
//

#include "JSV8RemoteConnector.h"

using namespace std;
using namespace dev;
using namespace dev::eth;

void JSV8RemoteConnector::onSend(char const* _payload)
{
	m_request.setUrl(m_url);
	m_request.setBody(_payload);
	long code;
	string response;
	tie(code, response) = m_request.post();
	(void)code;
	m_lastResponse = response.c_str();
}
