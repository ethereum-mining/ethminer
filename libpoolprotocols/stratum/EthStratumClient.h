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

class EthStratumClient : public PoolClient
{
public:

	typedef enum { STRATUM = 0, ETHPROXY, ETHEREUMSTRATUM } StratumProtocol;

	EthStratumClient(int const & worktimeout, string const & email, bool const & submitHashrate);
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
	void response_timeout_handler(const boost::system::error_code& ec);
	void hashrate_event_handler(const boost::system::error_code& ec);

	void reset_work_timeout();
	void readline();
	void handleResponse(const boost::system::error_code& ec);
	void handleHashrateResponse(const boost::system::error_code& ec);
	void readResponse(const boost::system::error_code& ec, std::size_t bytes_transferred);
	void processReponse(Json::Value& responseObject);
	void async_write_with_response();

	PoolConnection m_connection;

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
	boost::asio::ip::tcp::socket *m_socket;
	// Use shared ptrs to avoid crashes due to async_writes
	// see https://stackoverflow.com/questions/41526553/can-async-write-cause-segmentation-fault-when-this-is-deleted
	std::shared_ptr<boost::asio::ssl::stream<boost::asio::ip::tcp::socket> >
	  m_securesocket;
	std::shared_ptr<boost::asio::ip::tcp::socket>
	  m_nonsecuresocket;

	boost::asio::streambuf m_requestBuffer;
	boost::asio::streambuf m_responseBuffer;

	boost::asio::deadline_timer m_worktimer;
	boost::asio::deadline_timer m_responsetimer;
	boost::asio::deadline_timer m_hashrate_event;
	bool m_response_pending = false;

	boost::asio::ip::tcp::resolver m_resolver;

	string m_email;
	string m_rate;

	double m_nextWorkDifficulty;

	h64 m_extraNonce;
	int m_extraNonceHexSize;
	
	bool m_submit_hashrate = false;
	string m_submit_hashrate_id;

	void processExtranonce(std::string& enonce);

	bool m_linkdown = true;
};
