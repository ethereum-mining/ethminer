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

	EthStratumClient(int worktimeout, int responsetimeout, string const & email, bool const & submitHashrate);
	~EthStratumClient();

	void connect();
	void disconnect();
	
	// Connected and Connection Statuses
	bool isConnected()
	{
		return m_connected.load(std::memory_order_relaxed) &&
				!m_disconnecting.load(std::memory_order_relaxed);
	}
	bool isSubscribed() { return m_subscribed.load(std::memory_order_relaxed); }
	bool isAuthorized() { return m_authorized.load(std::memory_order_relaxed); }
	string ActiveEndPoint() { return " [" + toString(m_endpoint) + "]"; };

	void submitHashrate(string const & rate);
	void submitSolution(Solution solution);

	h256 currentHeaderHash() { return m_current.header; }
	bool current() { return static_cast<bool>(m_current); }

private:

	void resolve_handler(const boost::system::error_code& ec, boost::asio::ip::tcp::resolver::iterator i);
	void start_connect(boost::asio::ip::tcp::resolver::iterator endpoint_iter);
	void check_connect_timeout(const boost::system::error_code& ec);
	void connect_handler(const boost::system::error_code& ec, boost::asio::ip::tcp::resolver::iterator i);
	void work_timeout_handler(const boost::system::error_code& ec);
	void response_timeout_handler(const boost::system::error_code& ec);

	void reset_work_timeout();
	void processReponse(Json::Value& responseObject);
	std::string processError(Json::Value& erroresponseObject);
	void processExtranonce(std::string& enonce);

	void recvSocketData();
	void onRecvSocketDataCompleted(const boost::system::error_code& ec, std::size_t bytes_transferred);
	void sendSocketData(Json::Value const & jReq);
	void onSendSocketDataCompleted(const boost::system::error_code& ec);


	string m_worker; // eth-proxy only; No ! It's for all !!!

	std::atomic<bool> m_subscribed = { false };
	std::atomic<bool> m_authorized = { false };
	std::atomic<bool> m_connected = { false };
	std::atomic<bool> m_disconnecting = { false };

	// seconds to trigger a work_timeout (overwritten in constructor)
	int m_worktimeout;
	
	// seconds timeout for responses and connection (overwritten in constructor)
	int m_responsetimeout;

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

	boost::asio::streambuf m_sendBuffer;
	boost::asio::streambuf m_recvBuffer;
	Json::FastWriter m_jWriter;

	boost::asio::deadline_timer m_conntimer;
	boost::asio::deadline_timer m_worktimer;
	boost::asio::deadline_timer m_responsetimer;
	bool m_response_pending = false;

	boost::asio::ip::tcp::resolver m_resolver;

	string m_email;
	string m_rate;

	double m_nextWorkDifficulty;

	h64 m_extraNonce;
	int m_extraNonceHexSize;
	
	bool m_submit_hashrate;
	std::string m_submit_hashrate_id;

};
