#include "Api.h"

Api::Api(const unsigned int &port, Farm &farm): m_connection(TcpSocketServer("0.0.0.0", port)), m_farm(farm)
{
	if (port > 0) {
		this->m_server = new ApiServer(this->m_connection, JSONRPC_SERVER_V1V2, this->m_farm);
		this->m_server->StartListening();
	}
}