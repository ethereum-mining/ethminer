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
using namespace boost::asio;
using boost::asio::ip::tcp;
using namespace dev;
using namespace dev::eth;

typedef struct {
	string host;
	string port;
	string user;
	string pass;
} cred_t;

class EthStratumClient
{
public:
	EthStratumClient(GenericFarm<EthashProofOfWork> * f, MinerType m, string const & host, string const & port, string const & user, string const & pass, int const & retries, bool const & precompute);
	~EthStratumClient();

	void setFailover(string const & host, string const & port);
	void setFailover(string const & host, string const & port, string const & user, string const & pass);

	bool isRunning() { return m_running; }
	bool isConnected() { return m_connected; }
	h256 currentHeaderHash() { return m_current.headerHash; }
	bool current() { return m_current; }
	bool submit(EthashProofOfWork::Solution solution);
private:
	void connect();
	void reconnect();
	void disconnect();
	void resolve_handler(const boost::system::error_code& ec, tcp::resolver::iterator i);
	void connect_handler(const boost::system::error_code& ec, tcp::resolver::iterator i);
	
	void readline();
	void handleResponse(const boost::system::error_code& ec);
	void readResponse(const boost::system::error_code& ec, std::size_t bytes_transferred);
	void processReponse(Json::Value& responseObject);
	
	MinerType m_minerType;

	cred_t * p_active;
	cred_t m_primary;
	cred_t m_failover;

	bool m_authorized;
	bool m_connected;
	bool m_precompute;
	bool m_running = true;

	int	m_retries = 0;
	int	m_maxRetries;

	boost::mutex m_mtx;
	int m_pending;
	string m_response;

	GenericFarm<EthashProofOfWork> * p_farm;
	EthashProofOfWork::WorkPackage m_current;
	string m_job;
	EthashAux::FullType m_dag;

	boost::asio::io_service m_io_service;
	tcp::socket m_socket;

	boost::asio::streambuf m_requestBuffer;
	boost::asio::streambuf m_responseBuffer;
};