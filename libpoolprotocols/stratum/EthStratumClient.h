#pragma once

#include <iostream>
#include <boost/array.hpp>
#include <boost/asio.hpp>
#include <boost/asio/ssl.hpp>
#include <boost/bind.hpp>
#include <json/json.h>
#include <libdevcore/Log.h>
#include <libdevcore/FixedHash.h>
#include <libethcore/Farm.h>
#include <libethcore/EthashAux.h>
#include <libethcore/Miner.h>
#include "../PoolClient.h"


using namespace std;
using namespace dev;
using namespace dev::eth;

enum class StratumSecure
{
	NONE,
	TLS12,
	TLS,
	ALLOW_SELFSIGNED
};


class EthStratumClient : public PoolClient
{
public:
	EthStratumClient(int const & worktimeout, int const & protocol, string const & email, bool const & submitHashrate, StratumSecure const & secureMode);
	~EthStratumClient();

	void connect();
	void disconnect();
	
	bool isConnected() { return m_connected.load(std::memory_order_relaxed) && m_authorized; }
	
	void submitHashrate(string const & rate);
	void submitSolution(Solution solution);

	h256 currentHeaderHash() { return m_current.header; }
	bool current() { return static_cast<bool>(m_current); }

private:

	void resolve_handler(const boost::system::error_code& ec, boost::asio::ip::tcp::resolver::iterator i);
	void connect_handler(const boost::system::error_code& ec, boost::asio::ip::tcp::resolver::iterator i);
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
	std::atomic<bool> m_connected = {false};

	int m_worktimeout = 60;

	std::mutex x_pending;
	int m_pending;

	WorkPackage m_current;

	bool m_stale = false;

	std::thread m_serviceThread;  ///< The IO service thread.
	boost::asio::io_service m_io_service;
	StratumSecure m_secureMode;
	boost::asio::ip::tcp::socket *m_socket;
	boost::asio::ssl::stream<boost::asio::ip::tcp::socket> *m_securesocket;

	boost::asio::streambuf m_requestBuffer;
	boost::asio::streambuf m_responseBuffer;

	boost::asio::deadline_timer m_worktimer;

	boost::asio::ip::tcp::resolver m_resolver;

	int m_protocol;
	string m_email;

	double m_nextWorkDifficulty;

	h64 m_extraNonce;
	int m_extraNonceHexSize;
	
	bool m_submit_hashrate = false;
	string m_submit_hashrate_id;

	void processExtranonce(std::string& enonce);
};
