#include <iostream>
#include <boost/array.hpp>
#include <boost/asio.hpp>
#include <boost/bind.hpp>
#include <json/json.h>
#include <libdevcore/Log.h>
#include <libdevcore/FixedHash.h>
#include <libdevcore/Worker.h>
#include <libethcore/Farm.h>
#include <libethcore/EthashAux.h>
#include <libethcore/Miner.h>
#include <libethcore/PoolClient.h>
#include "BuildInfo.h"


using namespace std;
using namespace boost::asio;
using boost::asio::ip::tcp;
using namespace dev;
using namespace dev::eth;

class EthStratumClientV2 : public Worker, public PoolClient
{
public:
	EthStratumClientV2(int const & protocol, string const & email, bool const & submitHashrate);
	~EthStratumClientV2();

	bool isRunning() { return m_running; }
	bool isConnected() { return m_connected && m_authorized; }
	h256 currentHeaderHash() { return m_current.header; }
	bool current() { return static_cast<bool>(m_current); }
	unsigned waitState() { return m_waitState; }

	void connect();
	void disconnect();

	void submitHashrate(string const & rate);
	void submitSolution(Solution solution);
private:
	void workLoop() override;

	void processReponse(Json::Value& responseObject);
	
	cred_t * p_active;
	cred_t m_primary;
	cred_t m_failover;

	string m_worker; // eth-proxy only;

	bool m_authorized;
	bool m_connected;
	bool m_running = true;

	int m_waitState = MINER_WAIT_STATE_WORK;

	string m_response;

	mutex x_current;
	WorkPackage m_current;
	WorkPackage m_previous;

	bool m_stale = false;

	string m_job;
	string m_previousJob;

	boost::asio::io_service m_io_service;
	tcp::socket m_socket;

	boost::asio::streambuf m_requestBuffer;
	boost::asio::streambuf m_responseBuffer;

	boost::asio::deadline_timer * p_worktimer;

	int m_protocol;
	string m_email;

	double m_nextWorkDifficulty;

	h64 m_extraNonce;
	int m_extraNonceHexSize;
	
	bool m_submit_hashrate = false;
	string m_submit_hashrate_id;

	void processExtranonce(std::string& enonce);
};