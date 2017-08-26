#include <iostream>
#include <boost/array.hpp>
#include <boost/asio.hpp>
#include <boost/bind.hpp>
#include <json/json.h>
#include <libdevcore/Log.h>
#include <libdevcore/FixedHash.h>
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


class EthStratumClient : public PoolClient
{
public:
	EthStratumClient(int const & worktimeout, int const & protocol, string const & email);
	~EthStratumClient();

	h256 currentHeaderHash() { return m_current.header; }
	bool current() { return static_cast<bool>(m_current); }
	
	void connect();
	void disconnect();

	bool isRunning() { return m_running; }
	bool isConnected() { return m_connected && m_authorized; }

	void submitHashrate(string const & rate);
	void submitSolution(Solution solution);

private:
	void resolve_handler(const boost::system::error_code& ec, tcp::resolver::iterator i);
	void connect_handler(const boost::system::error_code& ec, tcp::resolver::iterator i);
	void work_timeout_handler(const boost::system::error_code& ec);

	void readline();
	void handleResponse(const boost::system::error_code& ec);
	void readResponse(const boost::system::error_code& ec, std::size_t bytes_transferred);
	void processReponse(Json::Value& responseObject);

	cred_t * p_active;
	cred_t m_primary;
	cred_t m_failover;

	string m_worker; // eth-proxy only;

	bool m_authorized;
	bool m_connected;
	bool m_running = true;

	int m_worktimeout = 60;

	std::mutex x_pending;
	int m_pending;

	std::mutex x_current;
	WorkPackage m_current;
	WorkPackage m_previous;

	bool m_stale = false;

	string m_job;
	string m_previousJob;

	std::thread m_serviceThread;  ///< The IO service thread.
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
	
	string m_submit_hashrate_id;

	void processExtranonce(std::string& enonce);
};