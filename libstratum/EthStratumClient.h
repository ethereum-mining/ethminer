#include <iostream>
#include <boost/array.hpp>
#include <boost/asio.hpp>
#include <boost/bind.hpp>
#include <json/json.h>
#include <libdevcore/Log.h>

using namespace std;
using namespace boost::asio;
using boost::asio::ip::tcp;

class EthStratumClient
{
public:
	EthStratumClient(string const & host, string const & port, string const & user, string const & pass);
	~EthStratumClient();

	boost::asio::io_service m_io_service;
	
private:
	void connect();
	void resolve_handler(const boost::system::error_code& ec, tcp::resolver::iterator i);
	void connect_handler(const boost::system::error_code& ec, tcp::resolver::iterator i);
	
	void handleResponse(const boost::system::error_code& ec);
	void readResponse(const boost::system::error_code& ec, std::size_t bytes_transferred);


	void subscribe_handler(const boost::system::error_code& ec);
	void authorize_handler(const boost::system::error_code& ec);
	void read_subscribe_handler(const boost::system::error_code& ec, std::size_t bytes_transferred);
	void read_authorize_handler(const boost::system::error_code& ec, std::size_t bytes_transferred);
	void read_work_handler(const boost::system::error_code& ec, std::size_t bytes_transferred);

	string m_host;
	string m_port;
	string m_user;
	string m_pass;

	
	tcp::socket m_socket;

	boost::asio::streambuf m_requestBuffer;
	boost::asio::streambuf m_responseBuffer;
};