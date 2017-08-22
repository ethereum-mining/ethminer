#ifndef _APISERVER_H_
#define _APISERVER_H_

#include <libethcore/Farm.h>
#include <libethcore/Miner.h>
#include <jsonrpccpp/server.h>

using namespace jsonrpc;
using namespace dev;
using namespace dev::eth;

class ApiServer : public AbstractServer<ApiServer>
{
    public:
		ApiServer(AbstractServerConnector &conn, serverVersion_t type, Farm &farm);
	private:
		Farm &m_farm;
		void getMinerStat1(const Json::Value& request, Json::Value& response);
};

#endif //_APISERVER_H_