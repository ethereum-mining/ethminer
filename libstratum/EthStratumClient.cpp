 
#include "EthStratumClient.h"
#include <libdevcore/Log.h>
#include <libethash/endian.h>
using boost::asio::ip::tcp;


static void diffToTarget(uint32_t *target, double diff)
{
	uint32_t target2[8];
	uint64_t m;
	int k;

	for (k = 6; k > 0 && diff > 1.0; k--)
		diff /= 4294967296.0;
	m = (uint64_t)(4294901760.0 / diff);
	if (m == 0 && k == 6)
		memset(target2, 0xff, 32);
	else {
		memset(target2, 0, 32);
		target2[k] = (uint32_t)m;
		target2[k + 1] = (uint32_t)(m >> 32);
	}

	for (int i = 0; i < 32; i++)
		((uint8_t*)target)[31 - i] = ((uint8_t*)target2)[i];
}


EthStratumClient::EthStratumClient(Farm* f, MinerType m, string const & host, string const & port, string const & user, string const & pass, int const & retries, int const & worktimeout, int const & protocol, string const & email)
	: m_socket(m_io_service)
{
	m_minerType = m;
	m_primary.host = host;
	m_primary.port = port;
	m_primary.user = user;
	m_primary.pass = pass;

	p_active = &m_primary;

	m_authorized = false;
	m_connected = false;
	m_pending = 0;
	m_maxRetries = retries;
	m_worktimeout = worktimeout;

	m_protocol = protocol;
	m_email = email;

	m_submit_hashrate_id = h256::random().hex();
	
	p_farm = f;
	p_worktimer = nullptr;
	connect();
}

EthStratumClient::~EthStratumClient()
{
	m_io_service.stop();
	m_serviceThread.join();
}

void EthStratumClient::setFailover(string const & host, string const & port)
{
	setFailover(host, port, p_active->user, p_active->pass);
}

void EthStratumClient::setFailover(string const & host, string const & port, string const & user, string const & pass)
{
	m_failover.host = host;
	m_failover.port = port;
	m_failover.user = user;
	m_failover.pass = pass;
}

void EthStratumClient::connect()
{
	tcp::resolver r(m_io_service);
	tcp::resolver::query q(p_active->host, p_active->port);
	
	r.async_resolve(q, boost::bind(&EthStratumClient::resolve_handler,
					this, boost::asio::placeholders::error,
					boost::asio::placeholders::iterator));

	cnote << "Connecting to stratum server " << p_active->host + ":" + p_active->port;

	if (m_serviceThread.joinable())
	{
		// If the service thread have been created try to reset the service.
		m_io_service.reset();
	}
	else
	{
		// Otherwise, if the first time here, create new thread.
		m_serviceThread = std::thread{boost::bind(&boost::asio::io_service::run, &m_io_service)};
	}
}

#define BOOST_ASIO_ENABLE_CANCELIO 

void EthStratumClient::reconnect()
{
	if (p_worktimer) {
		p_worktimer->cancel();
		p_worktimer = nullptr;
	}

	m_io_service.reset();
	//m_socket.close(); // leads to crashes on Linux
	m_authorized = false;
	m_connected = false;
		
	if (!m_failover.host.empty())
	{
		m_retries++;

		if (m_retries > m_maxRetries)
		{
			if (m_failover.host == "exit") {
				disconnect();
				return;
			}
			else if (p_active == &m_primary)
			{
				p_active = &m_failover;
			}
			else {
				p_active = &m_primary;
			}
			m_retries = 0;
		}
	}
	
	cnote << "Reconnecting in 3 seconds...";
	boost::asio::deadline_timer timer(m_io_service, boost::posix_time::seconds(3));
	timer.wait();

	connect();
}

void EthStratumClient::disconnect()
{
	cnote << "Disconnecting";
	m_connected = false;
	m_running = false;
	if (p_farm->isMining())
	{
		cnote << "Stopping farm";
		p_farm->stop();
	}
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
		cerr << "Could not resolve host " << p_active->host + ":" + p_active->port + ", " << ec.message();
		reconnect();
	}
}

void EthStratumClient::connect_handler(const boost::system::error_code& ec, tcp::resolver::iterator i)
{
	dev::setThreadName("stratum");
	
	if (!ec)
	{
		m_connected = true;
		cnote << "Connected to stratum server " << i->host_name() << ":" << p_active->port;
		if (!p_farm->isMining())
		{
			cnote << "Starting farm";
			if (m_minerType == MinerType::CL)
				p_farm->start("opencl", false);
			else if (m_minerType == MinerType::CUDA)
				p_farm->start("cuda", false);
			else if (m_minerType == MinerType::Mixed) {
				p_farm->start("cuda", false);
				p_farm->start("opencl", true);
			}
		}
		std::ostream os(&m_requestBuffer);

		string user;
		size_t p;

		switch (m_protocol) {
			case STRATUM_PROTOCOL_STRATUM:
				os << "{\"id\": 1, \"method\": \"mining.subscribe\", \"params\": []}\n";
				break;
			case STRATUM_PROTOCOL_ETHPROXY:
				p = p_active->user.find_first_of(".");
				user = p_active->user.substr(0, p);
				if (p + 1 <= p_active->user.length())
					m_worker = p_active->user.substr(p + 1);
				else
					m_worker = "";

				if (m_email.empty())
				{
					os << "{\"id\": 1, \"worker\":\"" << m_worker << "\", \"method\": \"eth_submitLogin\", \"params\": [\"" << user << "\"]}\n";
				}
				else
				{
					os << "{\"id\": 1, \"worker\":\"" << m_worker << "\", \"method\": \"eth_submitLogin\", \"params\": [\"" << user << "\", \"" << m_email << "\"]}\n";
				}
				break;
			case STRATUM_PROTOCOL_ETHEREUMSTRATUM:
				os << "{\"id\": 1, \"method\": \"mining.subscribe\", \"params\": [\"ethminer/" << ETH_PROJECT_VERSION << "\",\"EthereumStratum/1.0.0\"]}\n";
				break;
		}
		
		async_write(m_socket, m_requestBuffer,
			boost::bind(&EthStratumClient::handleResponse, this,
									boost::asio::placeholders::error));
	}
	else
	{
		cwarn << "Could not connect to stratum server " << p_active->host << ":" << p_active->port << ", " << ec.message();
		reconnect();
	}

}

void EthStratumClient::readline() {
	x_pending.lock();
	if (m_pending == 0) {
		async_read_until(m_socket, m_responseBuffer, "\n",
			boost::bind(&EthStratumClient::readResponse, this,
			boost::asio::placeholders::error, boost::asio::placeholders::bytes_transferred));
	
		m_pending++;
		
	}
	x_pending.unlock();
}

void EthStratumClient::handleResponse(const boost::system::error_code& ec) {
	if (!ec)
	{
		readline();
	}
	else
	{
		dev::setThreadName("stratum");
		cwarn << "Handle response failed: " << ec.message();
	}
}

void EthStratumClient::readResponse(const boost::system::error_code& ec, std::size_t bytes_transferred)
{
	dev::setThreadName("stratum");
	x_pending.lock();
	m_pending = m_pending > 0 ? m_pending - 1 : 0;
	x_pending.unlock();

	if (!ec && bytes_transferred)
	{
		std::istream is(&m_responseBuffer);
		std::string response;
		getline(is, response);

		if (!response.empty() && response.front() == '{' && response.back() == '}') 
		{
			Json::Value responseObject;
			Json::Reader reader;
			if (reader.parse(response.c_str(), responseObject))
			{
				processReponse(responseObject);
			}
			else 
			{
				cwarn << "Parse response failed: " << reader.getFormattedErrorMessages();
			}
		}
		else if (m_protocol != STRATUM_PROTOCOL_ETHPROXY)
		{
			cwarn << "Discarding incomplete response";
		}
		if (m_connected)
			readline();
	}
	else
	{
		cwarn << "Read response failed: " << ec.message();
		if (m_connected)
			reconnect();
	}
}

void EthStratumClient::processExtranonce(std::string& enonce)
{
	m_extraNonceHexSize = enonce.length();

	cnote << "Extranonce set to " << enonce;

	for (int i = enonce.length(); i < 16; ++i) enonce += "0";
	m_extraNonce = h64(enonce);
}

void EthStratumClient::processReponse(Json::Value& responseObject)
{
	Json::Value error = responseObject.get("error", {});
	if (error.isArray())
	{
		string msg = error.get(1, "Unknown error").asString();
		cnote << msg;
	}
	std::ostream os(&m_requestBuffer);
	Json::Value params;
	int id = responseObject.get("id", Json::Value::null).asInt();
	switch (id)
	{
		case 1:
		if (m_protocol == STRATUM_PROTOCOL_ETHEREUMSTRATUM)
		{
			m_nextWorkDifficulty = 1;
			params = responseObject.get("result", Json::Value::null);
			if (params.isArray())
			{
				std::string enonce = params.get((Json::Value::ArrayIndex)1, "").asString();
				processExtranonce(enonce);
			}

			os << "{\"id\": 2, \"method\": \"mining.extranonce.subscribe\", \"params\": []}\n";
		}
		if (m_protocol != STRATUM_PROTOCOL_ETHPROXY)
		{
			cnote << "Subscribed to stratum server";
			os << "{\"id\": 3, \"method\": \"mining.authorize\", \"params\": [\"" << p_active->user << "\",\"" << p_active->pass << "\"]}\n";
		}
		else
		{
			m_authorized = true;
			os << "{\"id\": 5, \"method\": \"eth_getWork\", \"params\": []}\n"; // not strictly required but it does speed up initialization
		}
		async_write(m_socket, m_requestBuffer,
			boost::bind(&EthStratumClient::handleResponse, this,
			boost::asio::placeholders::error));
		break;
	case 2:
		// nothing to do...
		break;
	case 3:
		m_authorized = responseObject.get("result", Json::Value::null).asBool();
		if (!m_authorized)
		{
			cnote << "Worker not authorized:" << p_active->user;
			disconnect();
			return;
		}
		cnote << "Authorized worker " << p_active->user;
		break;
	case 4:
		if (responseObject.get("result", false).asBool()) {
			cnote << EthLime << "B-) Submitted and accepted." << EthReset;
			p_farm->acceptedSolution(m_stale);
		}
		else {
			cwarn << ":-( Not accepted.";
			p_farm->rejectedSolution(m_stale);
		}
		break;
	default:
		string method, workattr;
		unsigned index;
		if (m_protocol != STRATUM_PROTOCOL_ETHPROXY)
		{
			method = responseObject.get("method", "").asString();
			workattr = "params";
			index = 1;
		}
		else
		{
			method = "mining.notify";
			workattr = "result";
			index = 0;
		}

		if (method == "mining.notify")
		{
			params = responseObject.get(workattr.c_str(), Json::Value::null);
			if (params.isArray())
			{
				string job = params.get((Json::Value::ArrayIndex)0, "").asString();

				if (m_protocol == STRATUM_PROTOCOL_ETHEREUMSTRATUM)
				{
					string sSeedHash = params.get((Json::Value::ArrayIndex)1, "").asString();
					string sHeaderHash = params.get((Json::Value::ArrayIndex)2, "").asString();

					if (sHeaderHash != "" && sSeedHash != "")
					{
						cnote << "Received new job #" + job;

						h256 seedHash = h256(sSeedHash);

						m_previous.header = m_current.header;
						m_previous.seed = m_current.seed;
						m_previous.boundary = m_current.boundary;
						m_previous.startNonce = m_current.startNonce;
						m_previous.exSizeBits = m_previous.exSizeBits;
						m_previousJob = m_job;

						m_current.header = h256(sHeaderHash);
						m_current.seed = seedHash;
						m_current.boundary = h256();
						diffToTarget((uint32_t*)m_current.boundary.data(), m_nextWorkDifficulty);
						m_current.startNonce = ethash_swap_u64(*((uint64_t*)m_extraNonce.data()));
						m_current.exSizeBits = m_extraNonceHexSize * 4;
						m_job = job;

						p_farm->setWork(m_current);
					}
				}
				else
				{
					string sHeaderHash = params.get((Json::Value::ArrayIndex)index++, "").asString();
					string sSeedHash = params.get((Json::Value::ArrayIndex)index++, "").asString();
					string sShareTarget = params.get((Json::Value::ArrayIndex)index++, "").asString();

					// coinmine.pl fix
					int l = sShareTarget.length();
					if (l < 66)
						sShareTarget = "0x" + string(66 - l, '0') + sShareTarget.substr(2);


					if (sHeaderHash != "" && sSeedHash != "" && sShareTarget != "")
					{
						cnote << "Received new job #" + job.substr(0, 8);

						h256 seedHash = h256(sSeedHash);
						h256 headerHash = h256(sHeaderHash);

						if (headerHash != m_current.header)
						{
							//x_current.lock();
							if (p_worktimer)
								p_worktimer->cancel();

							m_previous.header = m_current.header;
							m_previous.seed = m_current.seed;
							m_previous.boundary = m_current.boundary;
							m_previousJob = m_job;

							m_current.header = h256(sHeaderHash);
							m_current.seed = seedHash;
							m_current.boundary = h256(sShareTarget);
							m_job = job;

							p_farm->setWork(m_current);
							//x_current.unlock();
							p_worktimer = new boost::asio::deadline_timer(m_io_service, boost::posix_time::seconds(m_worktimeout));
							p_worktimer->async_wait(boost::bind(&EthStratumClient::work_timeout_handler, this, boost::asio::placeholders::error));

						}
					}
				}
			}
		}
		else if (method == "mining.set_difficulty" && m_protocol == STRATUM_PROTOCOL_ETHEREUMSTRATUM)
		{
			params = responseObject.get("params", Json::Value::null);
			if (params.isArray())
			{
				m_nextWorkDifficulty = params.get((Json::Value::ArrayIndex)0, 1).asDouble();
				if (m_nextWorkDifficulty <= 0.0001) m_nextWorkDifficulty = 0.0001;
				cnote << "Difficulty set to " << m_nextWorkDifficulty;
			}
		}
		else if (method == "mining.set_extranonce" && m_protocol == STRATUM_PROTOCOL_ETHEREUMSTRATUM)
		{
			params = responseObject.get("params", Json::Value::null);
			if (params.isArray())
			{
				std::string enonce = params.get((Json::Value::ArrayIndex)0, "").asString();
				processExtranonce(enonce);
			}
		}
		else if (method == "client.get_version")
		{
			os << "{\"error\": null, \"id\" : " << id << ", \"result\" : \"" << ETH_PROJECT_VERSION << "\"}\n";
			async_write(m_socket, m_requestBuffer,
				boost::bind(&EthStratumClient::handleResponse, this,
				boost::asio::placeholders::error));
		}
		break;
	}

}

void EthStratumClient::work_timeout_handler(const boost::system::error_code& ec) {
	if (!ec) {
		cnote << "No new work received in" << m_worktimeout << "seconds.";
		reconnect();
	}
}

bool EthStratumClient::submitHashrate(string const & rate) {
	// There is no stratum method to submit the hashrate so we use the rpc variant.
	string json = "{\"id\": 6, \"jsonrpc\":\"2.0\", \"method\": \"eth_submitHashrate\", \"params\": [\"" + rate + "\",\"0x" + this->m_submit_hashrate_id + "\"]}\n";
	std::ostream os(&m_requestBuffer);
	os << json;
	write(m_socket, m_requestBuffer);
	return true;
}

bool EthStratumClient::submit(Solution solution) {
	x_current.lock();
	WorkPackage tempWork(m_current);
	string temp_job = m_job;
	WorkPackage tempPreviousWork(m_previous);
	string temp_previous_job = m_previousJob;
	x_current.unlock();

	cnote << "Solution found; Submitting to" << p_active->host << "...";

	string minernonce;
	string nonceHex = toHex(solution.nonce);
	if (m_protocol != STRATUM_PROTOCOL_ETHEREUMSTRATUM)
		cnote << "  Nonce:" << "0x" + nonceHex;
	else
		minernonce = nonceHex.substr(m_extraNonceHexSize, 16 - m_extraNonceHexSize);


	if (EthashAux::eval(tempWork.seed, tempWork.header, solution.nonce).value < tempWork.boundary)
	{
		string json;

		switch (m_protocol) {
			case STRATUM_PROTOCOL_STRATUM:
				json = "{\"id\": 4, \"method\": \"mining.submit\", \"params\": [\"" + p_active->user + "\",\"" + temp_job + "\",\"0x" + nonceHex + "\",\"0x" + tempWork.header.hex() + "\",\"0x" + solution.mixHash.hex() + "\"]}\n";
				break;
			case STRATUM_PROTOCOL_ETHPROXY:
				json = "{\"id\": 4, \"worker\":\"" + m_worker + "\", \"method\": \"eth_submitWork\", \"params\": [\"0x" + nonceHex + "\",\"0x" + tempWork.header.hex() + "\",\"0x" + solution.mixHash.hex() + "\"]}\n";
				break;
			case STRATUM_PROTOCOL_ETHEREUMSTRATUM:
				json = "{\"id\": 4, \"method\": \"mining.submit\", \"params\": [\"" + p_active->user + "\",\"" + temp_job + "\",\"" + minernonce + "\"]}\n";
				break;
		}

		std::ostream os(&m_requestBuffer);
		os << json;
		m_stale = false;
		async_write(m_socket, m_requestBuffer,
			boost::bind(&EthStratumClient::handleResponse, this,
			boost::asio::placeholders::error));
		return true;
	}
	else if (EthashAux::eval(tempPreviousWork.seed, tempPreviousWork.header, solution.nonce).value < tempPreviousWork.boundary)
	{
		string json;

		switch (m_protocol) {
		case STRATUM_PROTOCOL_STRATUM:
			json = "{\"id\": 4, \"method\": \"mining.submit\", \"params\": [\"" + p_active->user + "\",\"" + temp_previous_job + "\",\"0x" + nonceHex + "\",\"0x" + tempPreviousWork.header.hex() + "\",\"0x" + solution.mixHash.hex() + "\"]}\n";
			break;
		case STRATUM_PROTOCOL_ETHPROXY:
			json = "{\"id\": 4, \"worker\":\"" + m_worker + "\", \"method\": \"eth_submitWork\", \"params\": [\"0x" + nonceHex + "\",\"0x" + tempPreviousWork.header.hex() + "\",\"0x" + solution.mixHash.hex() + "\"]}\n";
			break;
		case STRATUM_PROTOCOL_ETHEREUMSTRATUM:
			json = "{\"id\": 4, \"method\": \"mining.submit\", \"params\": [\"" + p_active->user + "\",\"" + temp_previous_job + "\",\"" + minernonce + "\"]}\n";
			break;
		}

		std::ostream os(&m_requestBuffer);
		os << json;
		m_stale = true;
		cwarn << "Submitting stale solution.";
		async_write(m_socket, m_requestBuffer,
			boost::bind(&EthStratumClient::handleResponse, this,
			boost::asio::placeholders::error));
		return true;
	}
	else {
		m_stale = false;
		cwarn << "FAILURE: GPU gave incorrect result!";
		p_farm->failedSolution();
	}

	return false;
}

