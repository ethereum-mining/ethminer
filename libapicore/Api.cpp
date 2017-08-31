#include "Api.h"

Api::Api(const unsigned int &port, Farm &farm): m_farm(farm)
{
	if (port > 0) {
		TcpSocketServer *conn = new TcpSocketServer("0.0.0.0", port);
		this->m_server = new ApiServer(conn, JSONRPC_SERVER_V1V2, this->m_farm);
		this->m_server->StartListening();
	}
}