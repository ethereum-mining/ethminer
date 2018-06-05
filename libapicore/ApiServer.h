#pragma once

#include <libethcore/Farm.h>
#include <libethcore/Miner.h>
#include <json/json.h>
#include <boost/bind.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/asio.hpp>

using namespace dev;
using namespace dev::eth;
using namespace std::chrono;

using boost::asio::ip::tcp;

class ApiConnection
{
public:

	ApiConnection(boost::asio::io_service& io_service, int id, bool readonly, Farm& f) :
		m_sessionId(id),
		m_socket(io_service),
		m_io_strand(io_service),
		m_readonly(readonly),
		m_farm(f) {}

	~ApiConnection() {}

	void start();

	Json::Value getMinerStat1();
	Json::Value getMinerStatHR();

	using Disconnected = std::function<void(int const&)>;
	void onDisconnected(Disconnected const& _handler) { m_onDisconnected = _handler; }

	int getId() { return m_sessionId; }

	tcp::socket& socket() { return m_socket; }

private:

	void disconnect();
	void processRequest(Json::Value& requestObject);
	void recvSocketData();
	void onRecvSocketDataCompleted(const boost::system::error_code& ec, std::size_t bytes_transferred);
	void sendSocketData(Json::Value const & jReq);
	void onSendSocketDataCompleted(const boost::system::error_code& ec);

	Disconnected m_onDisconnected;

	int m_sessionId;

	tcp::socket m_socket;
	boost::asio::io_service::strand m_io_strand;
	boost::asio::streambuf m_sendBuffer;
	boost::asio::streambuf m_recvBuffer;
	Json::FastWriter m_jWriter;

	bool m_readonly =  false ;
	Farm& m_farm;

};


class ApiServer
{
public:

	ApiServer(boost::asio::io_service& io_service, int portnum, bool readonly, Farm& f);
	bool isRunning() { return m_running.load(std::memory_order_relaxed); };
	void start();
	void stop();

private:

	void begin_accept();
	void handle_accept(std::shared_ptr<ApiConnection> session, boost::system::error_code ec);

	int lastSessionId = 0;

	std::thread m_workThread;
	std::atomic<bool> m_readonly = { false };
	std::atomic<bool> m_running = { false };
	int m_portnumber;
	tcp::acceptor m_acceptor;
	boost::asio::io_service::strand m_io_strand;
	std::vector<std::shared_ptr<ApiConnection>> m_sessions;

	Farm& m_farm;

};
