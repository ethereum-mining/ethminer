#ifndef ETH_GETWORK_CLIENT_H_
#define ETH_GETWORK_CLIENT_H_

#include <jsonrpccpp/client/connectors/httpclient.h>
#include <libethcore/PoolClient.h>
#include <iostream>
#include <boost/asio.hpp>
#include <boost/bind.hpp>
#include "jsonrpc_getwork.h"

using namespace std;
using namespace boost::asio;
using namespace dev;
using namespace eth;

class EthGetworkClient : public PoolClient
{
public:
	EthGetworkClient(unsigned const & farmRecheckPeriod);
	~EthGetworkClient();

	void connect();
	void disconnect();

	bool isRunning() { return m_running; }
	bool isConnected() { return m_connected; }

	void submitHashrate(string const & rate);
	void submitSolution(Solution solution);

private:
	bool m_justConnected = false;
	h256 m_client_id;
	JsonrpcGetwork *p_client;
	WorkPackage m_prevWorkPackage;

	std::thread m_serviceThread;  ///< The IO service thread.
	boost::asio::io_service m_io_service;
	unsigned m_farmRecheckPeriod = 500;
	boost::asio::deadline_timer * p_worktimer;

	void getWorkHandler(const boost::system::error_code& ec);
};

#endif