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

#include "BuildInfo.h"


using namespace std;
using namespace dev;
using namespace dev::eth;

class EthStratumClientV2 : public Worker
{
public:
	EthStratumClientV2(Farm* f, MinerType m, string const & host, string const & port, string const & user, string const & pass, int const & retries, int const & worktimeout, int const & protocol, string const & email);
	~EthStratumClientV2();

	void setFailover(string const & host, string const & port);
	void setFailover(string const & host, string const & port, string const & user, string const & pass);

	bool isRunning() { return m_running; }
	bool isConnected() { return m_connected && m_authorized; }
	h256 currentHeaderHash() { return m_current.header; }
	bool current() { return static_cast<bool>(m_current); }
	unsigned waitState() { return m_waitState; }
	bool submitHashrate(string const & rate);
	bool submit(Solution solution);
	void reconnect();
private:
	void workLoop() override;
	void connect();
	
	void disconnect();
	void work_timeout_handler(const boost::system::error_code& ec);

	void processReponse(Json::Value& responseObject);
	
	MinerType m_minerType;

	cred_t * p_active;
	cred_t m_primary;
	cred_t m_failover;

	string m_worker; // eth-proxy only;

	bool m_authorized;
	bool m_connected;
	bool m_running = true;

	int	m_retries = 0;
	int	m_maxRetries;
	int m_worktimeout = 60;

	int m_waitState = MINER_WAIT_STATE_WORK;

	string m_response;

	Farm* p_farm;
	mutex x_current;
	WorkPackage m_current;
	WorkPackage m_previous;

	bool m_stale = false;

	string m_job;
	string m_previousJob;

	boost::asio::io_service m_io_service;
	boost::asio::ip::tcp::socket m_socket;

	boost::asio::streambuf m_requestBuffer;
	boost::asio::streambuf m_responseBuffer;

	boost::asio::deadline_timer * p_worktimer;

	int m_protocol;
	string m_email;

	double m_nextWorkDifficulty;

	h64 m_extraNonce;
	int m_extraNonceHexSize;
	
	string m_submit_hashrate_id;

	void processExtranonce(std::string& enonce);
};