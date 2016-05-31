
#include "EthStratumClientV2.h"
#include <libdevcore/Log.h>
using boost::asio::ip::tcp;


EthStratumClientV2::EthStratumClientV2(GenericFarm<EthashProofOfWork> * f, MinerType m, string const & host, string const & port, string const & user, string const & pass, int const & retries, int const & worktimeout)
	: Worker("stratum"), 
	  m_socket(m_io_service)
{
	m_minerType = m;
	m_primary.host = host;
	m_primary.port = port;
	m_primary.user = user;
	m_primary.pass = pass;

	p_active = &m_primary;

	m_authorized = false;
	m_connected = false;
	m_maxRetries = retries;
	m_worktimeout = worktimeout;

	p_farm = f;
	p_worktimer = nullptr;
	startWorking();
}

EthStratumClientV2::~EthStratumClientV2()
{

}

void EthStratumClientV2::setFailover(string const & host, string const & port)
{
	setFailover(host, port, p_active->user, p_active->pass);
}

void EthStratumClientV2::setFailover(string const & host, string const & port, string const & user, string const & pass)
{
	m_failover.host = host;
	m_failover.port = port;
	m_failover.user = user;
	m_failover.pass = pass;
}

void EthStratumClientV2::workLoop() 
{
	while (m_running)
	{
		try {
			if (!m_connected)
			{
				m_io_service.run();
				connect();
			}
			read_until(m_socket, m_responseBuffer, "\n");
			std::istream is(&m_responseBuffer);
			std::string response;
			getline(is, response);

			if (response.front() == '{' && response.back() == '}')
			{
				Json::Value responseObject;
				Json::Reader reader;
				if (reader.parse(response.c_str(), responseObject))
				{
					processReponse(responseObject);
					m_response = response;
				}
				else
				{
					cwarn << "Parse response failed: " << reader.getFormattedErrorMessages();
				}
			}
			else
			{
				cwarn << "Discarding incomplete response";
			}
		}
		catch (std::exception const& _e) {
			cwarn << _e.what();
			reconnect();
		}
	}
}


void EthStratumClientV2::connect()
{
	cnote << "Connecting to stratum server " << p_active->host + ":" + p_active->port;

	tcp::resolver r(m_io_service);
	tcp::resolver::query q(p_active->host, p_active->port);
	tcp::resolver::iterator endpoint_iterator = r.resolve(q);
	tcp::resolver::iterator end;

	boost::system::error_code error = boost::asio::error::host_not_found;
	while (error && endpoint_iterator != end)
	{
		m_socket.close();
		m_socket.connect(*endpoint_iterator++, error);
	}
	if (error)
	{
		cerr << "Could not connect to stratum server " << p_active->host + ":" + p_active->port + ", " << error.message();
		reconnect();
	}
	else
	{
		cnote << "Connected!";
		m_connected = true;
		if (!p_farm->isMining())
		{
			cnote << "Starting farm";
			if (m_minerType == MinerType::CPU)
				p_farm->start("cpu");
			else if (m_minerType == MinerType::CL)
				p_farm->start("opencl");
			else if (m_minerType == MinerType::CUDA)
				p_farm->start("cuda");
		}
		std::ostream os(&m_requestBuffer);
		os << "{\"id\": 1, \"method\": \"mining.subscribe\", \"params\": []}\n";
		write(m_socket, m_requestBuffer);
	}
}

#define BOOST_ASIO_ENABLE_CANCELIO 

void EthStratumClientV2::reconnect()
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
	boost::asio::deadline_timer     timer(m_io_service, boost::posix_time::seconds(3));
	timer.wait();
}

void EthStratumClientV2::disconnect()
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

void EthStratumClientV2::processReponse(Json::Value& responseObject)
{
	Json::Value error = responseObject.get("error", new Json::Value);
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
		cnote << "Subscribed to stratum server";

		os << "{\"id\": 2, \"method\": \"mining.authorize\", \"params\": [\"" << p_active->user << "\",\"" << p_active->pass << "\"]}\n";

		write(m_socket, m_requestBuffer);
		break;
	case 2:
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
			cnote << "B-) Submitted and accepted.";
			p_farm->acceptedSolution(m_stale);
		}
		else {
			cwarn << ":-( Not accepted.";
			p_farm->rejectedSolution(m_stale);
		}
		break;
	default:
		string method = responseObject.get("method", "").asString();
		if (method == "mining.notify")
		{
			params = responseObject.get("params", Json::Value::null);
			if (params.isArray())
			{
				string job = params.get((Json::Value::ArrayIndex)0, "").asString();
				string sHeaderHash = params.get((Json::Value::ArrayIndex)1, "").asString();
				string sSeedHash = params.get((Json::Value::ArrayIndex)2, "").asString();
				string sShareTarget = params.get((Json::Value::ArrayIndex)3, "").asString();
				//bool cleanJobs = params.get((Json::Value::ArrayIndex)4, "").asBool();
				
				// coinmine.pl fix
				int l = sShareTarget.length();
				if (l < 66)
					sShareTarget = "0x" + string(66 - l, '0') + sShareTarget.substr(2);
								

				if (sHeaderHash != "" && sSeedHash != "" && sShareTarget != "")
				{
					cnote << "Received new job #" + job.substr(0,8);

					h256 seedHash = h256(sSeedHash);
					h256 headerHash = h256(sHeaderHash);

					if (headerHash != m_current.headerHash)
					{
						//x_current.lock();
						if (p_worktimer)
							p_worktimer->cancel();

						m_previous.headerHash = m_current.headerHash;
						m_previous.seedHash = m_current.seedHash;
						m_previous.boundary = m_current.boundary;
						m_previousJob = m_job;

						m_current.headerHash = h256(sHeaderHash);
						m_current.seedHash = seedHash;
						m_current.boundary = h256(sShareTarget);
						m_job = job;

						p_farm->setWork(m_current);
						//x_current.unlock();
						p_worktimer = new boost::asio::deadline_timer(m_io_service, boost::posix_time::seconds(m_worktimeout));
						p_worktimer->async_wait(boost::bind(&EthStratumClientV2::work_timeout_handler, this, boost::asio::placeholders::error));

					}
				}
			}
		}
		else if (method == "mining.set_difficulty")
		{

		}
		else if (method == "client.get_version")
		{
			os << "{\"error\": null, \"id\" : " << id << ", \"result\" : \"" << ETH_PROJECT_VERSION << "\"}\n";
			write(m_socket, m_requestBuffer);
		}
		break;
	}

}

void EthStratumClientV2::work_timeout_handler(const boost::system::error_code& ec) {
	if (!ec) {
		cnote << "No new work received in" << m_worktimeout << "seconds.";
		reconnect();
	}
}

bool EthStratumClientV2::submit(EthashProofOfWork::Solution solution) {
	x_current.lock();
	EthashProofOfWork::WorkPackage tempWork(m_current);
	string temp_job = m_job;
	EthashProofOfWork::WorkPackage tempPreviousWork(m_previous);
	string temp_previous_job = m_previousJob;
	x_current.unlock();

	cnote << "Solution found; Submitting to" << p_active->host << "...";
	cnote << "  Nonce:" << "0x" + solution.nonce.hex();

	if (EthashAux::eval(tempWork.seedHash, tempWork.headerHash, solution.nonce).value < tempWork.boundary)
	{
		string json = "{\"id\": 4, \"method\": \"mining.submit\", \"params\": [\"" + p_active->user + "\",\"" + temp_job + "\",\"0x" + solution.nonce.hex() + "\",\"0x" + tempWork.headerHash.hex() + "\",\"0x" + solution.mixHash.hex() + "\"]}\n";
		std::ostream os(&m_requestBuffer);
		os << json;
		m_stale = false;
		write(m_socket, m_requestBuffer);
		return true;
	}
	else if (EthashAux::eval(tempPreviousWork.seedHash, tempPreviousWork.headerHash, solution.nonce).value < tempPreviousWork.boundary)
	{
		string json = "{\"id\": 4, \"method\": \"mining.submit\", \"params\": [\"" + p_active->user + "\",\"" + temp_previous_job + "\",\"0x" + solution.nonce.hex() + "\",\"0x" + tempPreviousWork.headerHash.hex() + "\",\"0x" + solution.mixHash.hex() + "\"]}\n";
		std::ostream os(&m_requestBuffer);
		os << json;
		m_stale = true;
		cwarn << "Submitting stale solution.";
		write(m_socket, m_requestBuffer);
		return true;
	}
	else {
		m_stale = false;
		cwarn << "FAILURE: GPU gave incorrect result!";
		p_farm->failedSolution();
	}

	return false;
}

