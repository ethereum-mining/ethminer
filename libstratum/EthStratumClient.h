#include <iostream>
#include <boost/array.hpp>
#include <boost/asio.hpp>
#include <boost/bind.hpp>
#include <json/json.h>
#include <libdevcore/Log.h>
#include <libdevcore/FixedHash.h>
#include <libethcore/Farm.h>
#include <libethcore/EthashAux.h>
#include "BuildInfo.h"


using namespace std;
using namespace boost::asio;
using boost::asio::ip::tcp;
using namespace dev;
using namespace dev::eth;

class EthStratumClient
{
public:
	EthStratumClient(GenericFarm<EthashProofOfWork> * f, string const & host, string const & port, string const & user, string const & pass);
	~EthStratumClient();

	bool isRunning() { return m_running; }
	h256 currentHeaderHash() { return m_current.headerHash; }
	bool current() { return m_current; }
	bool submit(EthashProofOfWork::Solution solution);
private:
	void connect();
	void disconnect();
	void resolve_handler(const boost::system::error_code& ec, tcp::resolver::iterator i);
	void connect_handler(const boost::system::error_code& ec, tcp::resolver::iterator i);
	
	void handleResponse(const boost::system::error_code& ec);
	void readResponse(const boost::system::error_code& ec, std::size_t bytes_transferred);

	

	string m_host;
	string m_port;
	string m_user;
	string m_pass;
	bool   m_authorized;
	bool   m_running;
	bool   m_precompute;

	GenericFarm<EthashProofOfWork> * p_farm;
	EthashProofOfWork::WorkPackage m_current;
	string m_job;
	EthashAux::FullType m_dag;

	boost::asio::io_service m_io_service;
	tcp::socket m_socket;

	boost::asio::streambuf m_requestBuffer;
	boost::asio::streambuf m_responseBuffer;
};