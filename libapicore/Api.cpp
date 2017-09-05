#include "Api.h"

Api::Api(const int &port, Farm &farm): m_farm(farm)
{
	int portNumber = port;
	bool readonly = true;

	// > 0 = rw, < 0 = ro, 0 = disabled
	if (portNumber > 0) {
		readonly = false;
	} else if(portNumber < 0) {
		portNumber = -portNumber;
	}

	if (portNumber > 0) {
		TcpSocketServer *conn = new TcpSocketServer("0.0.0.0", portNumber);
		this->m_server = new ApiServer(conn, JSONRPC_SERVER_V2, this->m_farm, readonly);
		this->m_server->StartListening();
	}
}