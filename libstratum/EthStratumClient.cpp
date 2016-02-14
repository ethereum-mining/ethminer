
#include "EthStratumClient.h"

using boost::asio::ip::tcp;


EthStratumClient::EthStratumClient(GenericFarm<EthashProofOfWork> * f, string const & host, string const & port, string const & user, string const & pass)
	: m_socket(m_io_service)
{
	m_host = host;
	m_port = port;
	m_user = user;
	m_pass = pass;
	m_authorized = false;
	m_running = true;
	m_precompute = true;
	p_farm = f;
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

void EthStratumClient::disconnect()
{
	cnote << "Disconnecting";
	m_running = false;
	m_socket.close();
	m_io_service.stop();
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
		cerr << "Could not resolve host" << m_host + ":" + m_port + ", " << ec.message();
		disconnect();
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
		cwarn << "Could not connect to stratum server " << m_host << ":" << m_port << ", " << ec.message();
		disconnect();
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
		cwarn << "Handle response failed: " << ec.message();
	}
}

void EthStratumClient::readResponse(const boost::system::error_code& ec, std::size_t bytes_transferred)
{
	if (!ec)
	{
		std::istream is(&m_responseBuffer);
		std::string response;
		getline(is, response);
		//cnote << response;

		Json::Value responseObject;
		Json::Reader reader;
		if (reader.parse(response.c_str(), responseObject))
		{
			Json::Value error = responseObject.get("error", NULL);
			if (error.isArray())
			{
				string msg = error.get(1, "Unknown error").asString();
				cnote << msg;
			}
			std::ostream os(&m_requestBuffer);
			Json::Value params;
			int id = responseObject.get("id", NULL).asInt();	
			switch (id) 
			{
				case 1:
					cnote << "Subscribed to stratum server";

					os << "{\"id\": 2, \"method\": \"mining.authorize\", \"params\": [\"" << m_user << "\",\"" << m_pass << "\"]}\n";

					async_write(m_socket, m_requestBuffer,
						boost::bind(&EthStratumClient::handleResponse, this,
						boost::asio::placeholders::error));
					break;
				case 2:
					m_authorized = responseObject.get("result", NULL).asBool();
					if (!m_authorized)
					{
						disconnect();
						return;
					}
					cnote << "Authorized worker " << m_user;
					break;
				default:
					string method = responseObject.get("method", "").asString();
					if (method == "mining.notify")
					{
						params = responseObject.get("params", NULL);
						if (params.isArray())
						{
							string jobNumber	= params.get((Json::Value::ArrayIndex)0, "").asString();
							string sHeaderHash	= params.get((Json::Value::ArrayIndex)1, "").asString();
							string sSeedHash	= params.get((Json::Value::ArrayIndex)2, "").asString();
							string sShareTarget	= params.get((Json::Value::ArrayIndex)3, "").asString();
							bool cleanJobs		= params.get((Json::Value::ArrayIndex)4, "").asBool();
							if (sHeaderHash != "" && sSeedHash != "" && sShareTarget != "")
							{
								cnote << "Received new job #" + jobNumber;
								cnote << "Header hash: 0x" + sHeaderHash;
								cnote << "Seed hash: 0x" + sSeedHash;
								cnote << "Share target: 0x" + sShareTarget;

								h256 seedHash = h256(sSeedHash);
								h256 headerHash = h256(sHeaderHash);
								EthashAux::FullType dag;


								if (seedHash != m_current.seedHash)
									cnote << "Grabbing DAG for" << seedHash;
								if (!(dag = EthashAux::full(seedHash, true, [&](unsigned _pc){ cout << "\rCreating DAG. " << _pc << "% done..." << flush; return 0; })))
									BOOST_THROW_EXCEPTION(DAGCreationFailure());
								if (m_precompute)
									EthashAux::computeFull(sha3(seedHash), true);
								if (headerHash != m_current.headerHash)
								{
									m_current.headerHash = h256(sHeaderHash);
									m_current.seedHash = seedHash;
									m_current.boundary = h256(sShareTarget, h256::AlignRight);
									p_farm->setWork(m_current);
								}
							}
						}
					}
					else if (method == "mining.set_difficulty")
					{

					}
					else if (method == "client.get_version")
					{
						os << "{\"error\": null, \"id\" : "<< id <<", \"result\" : \"" << ETH_PROJECT_VERSION << "\"}";
						async_write(m_socket, m_requestBuffer,
							boost::bind(&EthStratumClient::handleResponse, this,
							boost::asio::placeholders::error));
					}
					break;
			}

			async_read_until(m_socket, m_responseBuffer, "\n",
				boost::bind(&EthStratumClient::readResponse, this,
				boost::asio::placeholders::error, boost::asio::placeholders::bytes_transferred));

		}
		else
		{
			cwarn << "Parse response failed: " << reader.getFormattedErrorMessages();
		}
	}
	else
	{
		cwarn << "Read response failed: " << ec.message();
	}
}

