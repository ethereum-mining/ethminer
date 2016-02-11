#include "EthStratumClient.h"

using boost::asio::ip::tcp;

EthStratumClient::EthStratumClient(string const & host, string const & port, string const & user, string const & pass)
	: m_socket(m_io_service)
{
	m_host = host;
	m_port = port;
	m_user = user;
	m_pass = pass;

	connect();


}

EthStratumClient::~EthStratumClient()
{

}

void EthStratumClient::connect()
{
	
	tcp::resolver r(m_io_service);
	tcp::resolver::query q(m_host, m_port);
	
	r.async_resolve(q, boost::bind(&EthStratumClient::resolve_handler,
																	this, boost::asio::placeholders::error,
																	boost::asio::placeholders::iterator));

	boost::thread t(boost::bind(&boost::asio::io_service::run, &m_io_service));
	
}

void EthStratumClient::resolve_handler(const boost::system::error_code& ec, tcp::resolver::iterator i)
{
	if (!ec)
	{
		async_connect(m_socket, i, boost::bind(&EthStratumClient::connect_handler,
																					this, boost::asio::placeholders::error,
																					boost::asio::placeholders::iterator));
	}
	else
	{
		cerr << "Could not resolve host " << m_host << ":" << m_port << ", " << ec.message();
	}
}

void EthStratumClient::connect_handler(const boost::system::error_code& ec, tcp::resolver::iterator i)
{
	if (!ec)
	{
		cnote << "Connected to stratum server " << m_host << ":" << m_port;

		std::ostream os(&m_requestBuffer);
		os << "{\"id\": 1, \"method\": \"mining.subscribe\", \"params\": []}\n";

		
		async_write(m_socket, m_requestBuffer,
			boost::bind(&EthStratumClient::handleResponse, this,
									boost::asio::placeholders::error));
	}
	else
	{
		cerr << "Could not connect to stratum server " << m_host << ":" << m_port << ", " << ec.message();
	}

}

void EthStratumClient::handleResponse(const boost::system::error_code& ec) {
	if (!ec)
	{
		async_read_until(m_socket, m_responseBuffer, "\n",
			boost::bind(&EthStratumClient::readResponse, this,
			boost::asio::placeholders::error, boost::asio::placeholders::bytes_transferred));

	}
	else
	{
		cerr << "Handle response failed: " << ec.message();
	}
}

void EthStratumClient::readResponse(const boost::system::error_code& ec, std::size_t bytes_transferred)
{
	if (!ec)
	{
		std::istream is(&m_responseBuffer);
		std::string response;
		getline(is, response);
		cnote << response;

		Json::Value responseObject;
		Json::Reader reader;
		if (reader.parse(response.c_str(), responseObject))
		{
			int error = responseObject.get("error", NULL).asInt();
			if (error != NULL)
			{
				cerr << "Error";
			}
			std::ostream os(&m_requestBuffer);
			int id = responseObject.get("id", NULL).asInt();	
			switch (id) 
			{
				case 1:
					cnote << "Subscribed";

					os << "{\"id\": 2, \"method\": \"mining.authorize\", \"params\": [\"" << m_user << "\",\"" << m_pass << "\"]}\n";

					//async_write(m_socket, m_requestBuffer,
					//	boost::bind(&EthStratumClient::handleResponse, this,
					//	boost::asio::placeholders::error));
					break;
				case 2:
					cnote << "Authorized";

					break;
				default:
					string method = responseObject.get("method", "").asString();
					cnote << method;
					break;
			}

			async_read_until(m_socket, m_responseBuffer, "\n",
				boost::bind(&EthStratumClient::readResponse, this,
				boost::asio::placeholders::error, boost::asio::placeholders::bytes_transferred));

		}
		else
		{
			cerr << "Parse response failed: " << reader.getFormattedErrorMessages();
		}
	}
	else
	{
		cerr << "Read response failed: " << ec.message();
	}
}

void EthStratumClient::subscribe_handler(const boost::system::error_code& ec) {
	if (!ec)
	{
		async_read_until(m_socket, m_responseBuffer, "\n",
			boost::bind(&EthStratumClient::read_subscribe_handler, this,
			boost::asio::placeholders::error, boost::asio::placeholders::bytes_transferred));

	}
	else
	{
		cerr << "Stratum subscription failed: "  << ec.message();
	}
}

void EthStratumClient::read_subscribe_handler(const boost::system::error_code& ec, std::size_t bytes_transferred)
{
	if (!ec)
	{
		std::istream is(&m_responseBuffer);
		std::string response;
		getline(is, response);
		cnote << response;

		std::ostream os(&m_requestBuffer);
		os << "{\"id\": 2, \"method\": \"mining.authorize\", \"params\": [\"" << m_user << "\",\"" << m_pass << "\"]}\n";

		//async_write(m_socket, m_requestBuffer,
		//	boost::bind(&EthStratumClient::authorize_handler, this,
		//	boost::asio::placeholders::error));

		async_read_until(m_socket, m_responseBuffer, "\n",
			boost::bind(&EthStratumClient::read_work_handler, this,
			boost::asio::placeholders::error, boost::asio::placeholders::bytes_transferred));
	}
	else
	{
		cerr << "Stratum read subscription failed: " << ec.message();
	}
}

void EthStratumClient::authorize_handler(const boost::system::error_code& ec) 
{
	if (!ec)
	{
		async_read_until(m_socket, m_responseBuffer, "\n",
			boost::bind(&EthStratumClient::read_authorize_handler, this,
			boost::asio::placeholders::error, boost::asio::placeholders::bytes_transferred));

	}
	else
	{
		cerr << "Stratum authorization failed: " << ec.message();
	}
}

void EthStratumClient::read_authorize_handler(const boost::system::error_code& ec, std::size_t bytes_transferred)
{
	if (!ec)
	{
		std::istream is(&m_responseBuffer);
		std::string response;
		getline(is, response);
		cnote << response;

		async_read_until(m_socket, m_responseBuffer, "\n",
			boost::bind(&EthStratumClient::read_work_handler, this,
			boost::asio::placeholders::error, boost::asio::placeholders::bytes_transferred));
	}
	else
	{
		cerr << "Stratum read authorization failed: " << ec.message();
	}
}

void EthStratumClient::read_work_handler(const boost::system::error_code& ec, std::size_t bytes_transferred)
{
	if (!ec)
	{
		std::istream is(&m_responseBuffer);
		std::string response;
		getline(is, response);
		cnote << response;

		async_read_until(m_socket, m_responseBuffer, "\n",
			boost::bind(&EthStratumClient::read_work_handler, this,
			boost::asio::placeholders::error, boost::asio::placeholders::bytes_transferred));
	}
	else
	{
		cerr << "Stratum read work failed: " << ec.message();
	}
}