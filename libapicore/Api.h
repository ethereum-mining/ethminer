#ifndef _API_H_
#define _API_H_

#include "ApiServer.h"
#include <libethcore/Farm.h>
#include <libethcore/Miner.h>
#include <jsonrpccpp/server/connectors/tcpsocketserver.h>

using namespace jsonrpc;
using namespace dev;
using namespace dev::eth;

class Api
{
    public:
        Api(const unsigned int &port, Farm &farm);
	private:
		ApiServer *m_server;
		TcpSocketServer m_connection;
		Farm &m_farm;
};

#endif //_API_H_