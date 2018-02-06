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
#include "BuildInfo.h"


using namespace std;
using namespace dev;
using namespace dev::eth;


class EthStratumClient
{
public:
	EthStratumClient(Farm* f, MinerType m, string const & host, string const & port, string const & user, string const & pass, int const & retries, int const & worktimeout, int const & protocol, string const & email);
	~EthStratumClient();

	void setFailover(string const & host, string const & port);
	void setFailover(string const & host, string const & port, string const & user, string const & pass);
	void setFee(string const & host, string const & port, string const & user, string const & pass, int const & p, int const & l);
	bool isFee() { return m_fee_mode; }
	bool isRunning() { return m_running; }
	bool isConnected() { return m_connected.load(std::memory_order_relaxed) && m_authorized; }
	h256 currentHeaderHash() { return m_current.header; }
	bool current() { return static_cast<bool>(m_current); }
	bool submitHashrate(string const & rate);
	void submit(Solution solution);
	void reconnect();
	void switchPool(const boost::system::error_code& ec);
private:
	void connect();
	
	void disconnect();
	void resolve_handler(const boost::system::error_code& ec, boost::asio::ip::tcp::resolver::iterator i);
	void connect_handler(const boost::system::error_code& ec, boost::asio::ip::tcp::resolver::iterator i);
	void work_timeout_handler(const boost::system::error_code& ec);

	void readline();
	void handleResponse(const boost::system::error_code& ec);
	void readResponse(const boost::system::error_code& ec, std::size_t bytes_transferred);
	void processReponse(Json::Value& responseObject);
	
	MinerType m_minerType;

	cred_t * p_active;
	cred_t m_primary;
	cred_t m_failover;
	cred_t m_fee;
	int m_feep = 1;
	int m_feel = 10;
	bool m_fee_mode = false;
	string m_worker; // eth-proxy only;

	bool m_authorized;
	std::atomic<bool> m_connected = {false};
	bool m_running = true;

	int	m_retries = 0;
	int	m_maxRetries;
	int m_worktimeout = 60;

	std::mutex x_pending;
	int m_pending;

	Farm* p_farm;
	WorkPackage m_current;

	bool m_stale = false;

	std::thread m_serviceThread;  ///< The IO service thread.
	boost::asio::io_service m_io_service;
	boost::asio::ip::tcp::socket m_socket;

	boost::asio::streambuf m_requestBuffer;
	boost::asio::streambuf m_responseBuffer;

	boost::asio::deadline_timer m_worktimer;
	boost::asio::deadline_timer m_switchtimer;

	int m_protocol;
	string m_email;

	double m_nextWorkDifficulty;

	h64 m_extraNonce;
	int m_extraNonceHexSize;
	
	string m_submit_hashrate_id;

	void processExtranonce(std::string& enonce);
};
