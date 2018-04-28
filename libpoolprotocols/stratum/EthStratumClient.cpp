#include "EthStratumClient.h"
#include <libdevcore/Log.h>
#include <libethash/endian.h>
#include <ethminer-buildinfo.h>

#ifdef _WIN32
#include <wincrypt.h>
#endif

using boost::asio::ip::tcp;

static uint64_t bswap(uint64_t val)
{
    val = ((val << 8) & 0xFF00FF00FF00FF00ULL) | ((val >> 8) & 0x00FF00FF00FF00FFULL);
    val = ((val << 16) & 0xFFFF0000FFFF0000ULL) | ((val >> 16) & 0x0000FFFF0000FFFFULL);
    return (val << 32) | (val >> 32);
}


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


EthStratumClient::EthStratumClient(int const & worktimeout, string const & email, bool const & submitHashrate) : PoolClient(),
	m_socket(nullptr),
	m_conntimer(m_io_service),
	m_worktimer(m_io_service),
	m_responsetimer(m_io_service),
	m_resolver(m_io_service)
{

	m_worktimeout = worktimeout;
	m_email = email;

	m_submit_hashrate = submitHashrate;
	if (m_submit_hashrate)
		m_submit_hashrate_id = h256::random().hex();
}

EthStratumClient::~EthStratumClient()
{
	m_io_service.stop();
	m_serviceThread.join();
}

void EthStratumClient::connect()
{

	m_connected.store(false, std::memory_order_relaxed);
	m_subscribed.store(false, std::memory_order_relaxed);
	m_authorized.store(false, std::memory_order_relaxed);


	// Prepare Socket

	if (m_conn.SecLevel() != SecureLevel::NONE) {

		boost::asio::ssl::context::method method = boost::asio::ssl::context::tls_client;
		if (m_conn.SecLevel() == SecureLevel::TLS12)
			method = boost::asio::ssl::context::tlsv12;

		boost::asio::ssl::context ctx(method);
		m_securesocket = std::make_shared<boost::asio::ssl::stream<boost::asio::ip::tcp::socket> >(m_io_service, ctx);
		m_socket = &m_securesocket->next_layer();

		if (m_conn.SecLevel() != SecureLevel::ALLOW_SELFSIGNED) {
			m_securesocket->set_verify_mode(boost::asio::ssl::verify_peer);

#ifdef _WIN32
			HCERTSTORE hStore = CertOpenSystemStore(0, "ROOT");
			if (hStore == NULL) {
				return;
			}

			X509_STORE *store = X509_STORE_new();
			PCCERT_CONTEXT pContext = NULL;
			while ((pContext = CertEnumCertificatesInStore(hStore, pContext)) != NULL) {
				X509 *x509 = d2i_X509(NULL,
					(const unsigned char **)&pContext->pbCertEncoded,
					pContext->cbCertEncoded);
				if (x509 != NULL) {
					X509_STORE_add_cert(store, x509);
					X509_free(x509);
				}
			}

			CertFreeCertificateContext(pContext);
			CertCloseStore(hStore, 0);

			SSL_CTX_set_cert_store(ctx.native_handle(), store);
#else
			char *certPath = getenv("SSL_CERT_FILE");
			try {
				ctx.load_verify_file(certPath ? certPath : "/etc/ssl/certs/ca-certificates.crt");
			}
			catch (...) {
				cwarn << "Failed to load ca certificates. Either the file '/etc/ssl/certs/ca-certificates.crt' does not exist";
				cwarn << "or the environment variable SSL_CERT_FILE is set to an invalid or inaccessable file.";
				cwarn << "It is possible that certificate verification can fail.";
			}
#endif
		}
	}
	else {
	  m_nonsecuresocket = std::make_shared<boost::asio::ip::tcp::socket>(m_io_service);
	  m_socket = m_nonsecuresocket.get();
	}

	// Activate keep alive to detect disconnects
	unsigned int keepAlive = 10000;

#if defined _WIN32 || defined WIN32 || defined OS_WIN64 || defined _WIN64 || defined WIN64 || defined WINNT
	int32_t timeout = keepAlive;
	setsockopt(m_socket->native_handle(), SOL_SOCKET, SO_RCVTIMEO, (const char*)&timeout, sizeof(timeout));
	setsockopt(m_socket->native_handle(), SOL_SOCKET, SO_SNDTIMEO, (const char*)&timeout, sizeof(timeout));
#else
	struct timeval tv;
	tv.tv_sec = keepAlive / 1000;
	tv.tv_usec = keepAlive % 1000;
	setsockopt(m_socket->native_handle(), SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
	setsockopt(m_socket->native_handle(), SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
#endif

	// Begin resolve and connect
	tcp::resolver::query q(m_conn.Host(), toString(m_conn.Port()));
	m_resolver.async_resolve(q,
		boost::bind(&EthStratumClient::resolve_handler,
			this, boost::asio::placeholders::error,
			boost::asio::placeholders::iterator));


	// IMPORTANT !!
	if (m_serviceThread.joinable())
	{
		// If the service thread have been created try to reset the service.
		m_io_service.reset();
	}
	else
	{
		// Otherwise, if the first time here, create new thread.
		m_serviceThread = std::thread{ boost::bind(&boost::asio::io_service::run, &m_io_service) };
	}



}

#define BOOST_ASIO_ENABLE_CANCELIO 

void EthStratumClient::disconnect()
{
	// Prevent unnecessary recursion
	if (m_disconnecting.load(std::memory_order::memory_order_relaxed)) {
		return;
	}
	else {
		m_disconnecting.store(true, std::memory_order::memory_order_relaxed);
	}

	m_conntimer.cancel();
	m_worktimer.cancel();
	m_responsetimer.cancel();
	m_response_pending = false;

	try {
		
		boost::system::error_code sec;

		if (m_conn.SecLevel() != SecureLevel::NONE) {
			m_securesocket->shutdown(sec);
		}
		else {
			m_nonsecuresocket->shutdown(boost::asio::ip::tcp::socket::shutdown_both, sec);
		}

		m_socket->close();
		m_io_service.stop();
	}
	catch (std::exception const& _e) {
		cwarn << "Error while disconnecting:" << _e.what();
	}

    if (m_securesocket) { m_securesocket = nullptr; }
	if (m_nonsecuresocket) { m_nonsecuresocket = nullptr; }
    if (m_socket) { m_socket = nullptr; }
	
	m_subscribed.store(false, std::memory_order_relaxed);
	m_authorized.store(false, std::memory_order_relaxed);

	if (m_onDisconnected) { m_onDisconnected();	}

	// Release locking flag and set connection status
	m_connected.store(false, std::memory_order_relaxed);
	m_disconnecting.store(false, std::memory_order::memory_order_relaxed);
}

void EthStratumClient::resolve_handler(const boost::system::error_code& ec, tcp::resolver::iterator i)
{
	dev::setThreadName("stratum");
	if (!ec)
	{

		// Start Connection Process and set timeout timer
		start_connect(i);
		m_conntimer.async_wait(boost::bind(&EthStratumClient::check_connect_timeout, this, boost::asio::placeholders::error));

	}
	else
	{
		cwarn << "Could not resolve host " << m_conn.Host() << ", " << ec.message();
		disconnect();
	}
}

void EthStratumClient::reset_work_timeout()
{
	m_worktimer.cancel();
	m_worktimer.expires_from_now(boost::posix_time::seconds(m_worktimeout));
	m_worktimer.async_wait(boost::bind(&EthStratumClient::work_timeout_handler, this, boost::asio::placeholders::error));
}

void EthStratumClient::start_connect(tcp::resolver::iterator endpoint_iter)
{
	if (endpoint_iter != tcp::resolver::iterator()) {

		cnote << ("Trying " + toString(endpoint_iter->endpoint()) + " ...");
		
		// Set timeout of 2 seconds
		m_conntimer.expires_from_now(boost::posix_time::seconds(2));

		// Start connecting async
		m_socket->async_connect(endpoint_iter->endpoint(), 
			boost::bind(&EthStratumClient::connect_handler, this, _1, endpoint_iter));

	}
	else {

		cwarn << "No more addresses to try !";
		disconnect();

	}
}

void EthStratumClient::check_connect_timeout(const boost::system::error_code& ec)
{
	(void)ec;

	// Check whether the deadline has passed. We compare the deadline against
	// the current time since a new asynchronous operation may have moved the
	// deadline before this actor had a chance to run.

	if (!isConnected()) {

		if (m_conntimer.expires_at() <= boost::asio::deadline_timer::traits_type::now())
		{
			// The deadline has passed. The socket is closed so that any outstanding
			// asynchronous operations are cancelled.
			m_socket->close();

			// There is no longer an active deadline. The expiry is set to positive
			// infinity so that the actor takes no action until a new deadline is set.
			m_conntimer.expires_at(boost::posix_time::pos_infin);
		}
		// Put the actor back to sleep.
		m_conntimer.async_wait(boost::bind(&EthStratumClient::check_connect_timeout, this, boost::asio::placeholders::error));
	}

}


void EthStratumClient::connect_handler(const boost::system::error_code& ec, tcp::resolver::iterator i)
{
	
	dev::setThreadName("stratum");

	// Timeout has run before
	if (!m_socket->is_open()) {

		cwarn << ("Error  " + toString((i)->endpoint()) + " [Timeout]");

		// Try the next available endpoint.
		start_connect(++i);

	} else if (ec) {

		cwarn << ("Error  " + toString((i)->endpoint()) + " [" + ec.message() + "]");
		
		// We need to close the socket used in the previous connection attempt
		// before starting a new one.
		// In case of error, in fact, boost does not close the socket
		m_socket->close();

		// Try the next available endpoint.
		start_connect(++i);

	}
	else {

		m_conntimer.cancel();
		m_endpoint = (i)->endpoint();


		if (m_conn.SecLevel() != SecureLevel::NONE) {

			boost::system::error_code hec;
			m_securesocket->handshake(boost::asio::ssl::stream_base::client, hec);

			if (hec) {
				cwarn << "SSL/TLS Handshake failed: " << hec.message();
				if (hec.value() == 337047686) { // certificate verification failed
					cwarn << "This can have multiple reasons:";
					cwarn << "* Root certs are either not installed or not found";
					cwarn << "* Pool uses a self-signed certificate";
					cwarn << "Possible fixes:";
					cwarn << "* Make sure the file '/etc/ssl/certs/ca-certificates.crt' exists and is accessible";
					cwarn << "* Export the correct path via 'export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt' to the correct file";
					cwarn << "  On most systems you can install the 'ca-certificates' package";
					cwarn << "  You can also get the latest file here: https://curl.haxx.se/docs/caextract.html";
					cwarn << "* Disable certificate verification all-together via command-line option.";
				}

				// Do not trigger a full disconnection but, instead, let the loop
				// continue with another IP (if any). 
				// Disconnection is triggered on no more IP available

				m_socket->close();
				start_connect(++i);
				return;
			}
		}

		// This should be done *after* a valid connection which may fail
		// on secure connection.
		m_connected.store(true, std::memory_order_relaxed);
		if (m_onConnected) { m_onConnected(); }

		// Successfully connected so we start our work timeout timer
		reset_work_timeout();

		string user;
		size_t p;

		switch (m_conn.Version()) {

			case EthStratumClient::STRATUM:

				sendSocketData("{\"id\": 1, \"method\": \"mining.subscribe\", \"params\": []}");
				break;

			case EthStratumClient::ETHPROXY:

				p = m_conn.User().find_first_of(".");
				user = m_conn.User().substr(0, p);
				if (p + 1 <= m_conn.User().length())
					m_worker = m_conn.User().substr(p + 1);
				else
					m_worker = "";

				if (m_email.empty())
				{
					sendSocketData("{\"id\": 1, \"worker\":\"" + m_worker + "\", \"method\": \"eth_submitLogin\", \"params\": [\"" + user + "\"]}");
				}
				else
				{
					sendSocketData("{\"id\": 1, \"worker\":\"" + m_worker + "\", \"method\": \"eth_submitLogin\", \"params\": [\"" + user + "\", \"" + m_email + "\"]}");
				}
				break;

			case EthStratumClient::ETHEREUMSTRATUM:

				sendSocketData( "{\"id\": 1, \"method\": \"mining.subscribe\", \"params\": [\"ethminer/" + toString(ethminer_get_buildinfo()->project_version) + "\",\"EthereumStratum/1.0.0\"]}");
				break;
		}

		// Begin receive data
		recvSocketData();

	}

}

void EthStratumClient::processExtranonce(std::string& enonce)
{
	m_extraNonceHexSize = enonce.length();

	cnote << "Extranonce set to " + enonce;

	for (int i = enonce.length(); i < 16; ++i) enonce += "0";
	m_extraNonce = h64(enonce);
}

void EthStratumClient::processReponse(Json::Value& responseObject)
{

	dev::setThreadName("stratum");

	Json::Value error = responseObject.get("error", {});
	if (error.isArray())
	{
		cnote << error.get(1, "Unknown error").asString();
	}

	Json::Value params;
	int id = responseObject.get("id", Json::Value::null).asInt();

	switch (id)
	{
		case 1:

			// Response to "mining.subscribe"
			switch (m_conn.Version()) {

			case EthStratumClient::STRATUM:

				m_subscribed.store(responseObject.get("result", Json::Value::null).asBool(), std::memory_order_relaxed);
				if (!m_subscribed)
				{
					cnote << "Could not subscribe to stratum server";
					disconnect();
					return;
				}

				cnote << "Subscribed to stratum server";
				sendSocketData("{\"id\": 3, \"method\": \"mining.authorize\", \"params\": [\"" + m_conn.User() + "\",\"" + m_conn.Pass() + "\"]}");
				break;

			case EthStratumClient::ETHPROXY:

				sendSocketData("{\"id\": 5, \"method\": \"eth_getWork\", \"params\": []}"); // not strictly required but it does speed up initialization
				break;

			case EthStratumClient::ETHEREUMSTRATUM:

				m_nextWorkDifficulty = 1;
				params = responseObject.get("result", Json::Value::null);
				if (params.isArray())
				{
					std::string enonce = params.get((Json::Value::ArrayIndex)1, "").asString();
					processExtranonce(enonce);
				}

				sendSocketData("{\"id\": 2, \"method\": \"mining.extranonce.subscribe\", \"params\": []}");
				break;
			}

		break;

	case 2:

		// nothing to do...
		break;

	case 3:

		// Response to "mining.authorize"
		m_authorized.store(responseObject.get("result", Json::Value::null).asBool(), std::memory_order_relaxed);
		if (!m_authorized)
		{
			cnote << "Worker not authorized:" + m_conn.User();
			disconnect();
			return;
		}
		cnote << "Authorized worker " + m_conn.User();
		break;

	case 4:

		// Response to solution submission
		{
			m_responsetimer.cancel();
			m_response_pending = false;
			if (responseObject.get("result", false).asBool()) {
				if (m_onSolutionAccepted) {
					m_onSolutionAccepted(m_stale);
				}
			}
			else {
				if (m_onSolutionRejected) {
					m_onSolutionRejected(m_stale);
				}
			}
		}
		break;


	default:

		string method, workattr;
		unsigned index;
		if (m_conn.Version() != EthStratumClient::ETHPROXY)
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
				if (m_response_pending)
					m_stale = true;
				if (m_conn.Version() == EthStratumClient::ETHEREUMSTRATUM)
				{
					string sSeedHash = params.get(1, "").asString();
					string sHeaderHash = params.get(2, "").asString();

					if (sHeaderHash != "" && sSeedHash != "")
					{
						reset_work_timeout();

                        m_current.header = h256(sHeaderHash);
						m_current.seed = h256(sSeedHash);
						m_current.boundary = h256();
						diffToTarget((uint32_t*)m_current.boundary.data(), m_nextWorkDifficulty);
						m_current.startNonce = bswap(*((uint64_t*)m_extraNonce.data()));
						m_current.exSizeBits = m_extraNonceHexSize * 4;
						m_current.job_len = job.size();
						if (m_conn.Version() == EthStratumClient::ETHEREUMSTRATUM)
							job.resize(64, '0');
						m_current.job = h256(job);

						if (m_onWorkReceived) {
							m_onWorkReceived(m_current);
						}
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
						h256 headerHash = h256(sHeaderHash);

						if (headerHash != m_current.header)
						{
							reset_work_timeout();

							m_current.header = h256(sHeaderHash);
							m_current.seed = h256(sSeedHash);
                            m_current.boundary = h256(sShareTarget);
							m_current.job = h256(job);

							if (m_onWorkReceived) {
								m_onWorkReceived(m_current);
							}
						}
					}
				}
			}
		}
		else if (method == "mining.set_difficulty" && m_conn.Version() == EthStratumClient::ETHEREUMSTRATUM)
		{
			params = responseObject.get("params", Json::Value::null);
			if (params.isArray())
			{
				m_nextWorkDifficulty = params.get((Json::Value::ArrayIndex)0, 1).asDouble();
				if (m_nextWorkDifficulty <= 0.0001) m_nextWorkDifficulty = 0.0001;
				cnote << "Difficulty set to "  << m_nextWorkDifficulty;
			}
		}
		else if (method == "mining.set_extranonce" && m_conn.Version() == EthStratumClient::ETHEREUMSTRATUM)
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
			sendSocketData("{\"error\": null, \"id\" : " + toString(id) + ", \"result\" : \"" + ethminer_get_buildinfo()->project_version + "\"})");
		}
		break;
	}

}

void EthStratumClient::work_timeout_handler(const boost::system::error_code& ec) {

	dev::setThreadName("stratum");

	if (!ec) {
		if (isConnected()) {
			cwarn << "No new work received in " << m_worktimeout << " seconds.";
			disconnect();
		}
	}

}

void EthStratumClient::response_timeout_handler(const boost::system::error_code& ec) {

	dev::setThreadName("stratum");

	if (!ec) {
		if (isConnected()) {
			cwarn << "No response received in 2 seconds.";
			disconnect();
		}
	}

}

void EthStratumClient::submitHashrate(string const & rate) {
	
	m_rate = rate;
	
	if (!m_submit_hashrate || !isConnected()) {
		return;
	}

	// There is no stratum method to submit the hashrate so we use the rpc variant.
	// Note !!
	// id = 6 is also the id used by ethermine.org and nanopool to push new jobs
	// thus we will be in trouble if we want to check the result of hashrate submission
	// actually change the id from 6 to 9

	string json = "{\"id\": 9, \"jsonrpc\":\"2.0\", \"method\": \"eth_submitHashrate\", \"params\": [\"" + m_rate + "\",\"0x" + toString(this->m_submit_hashrate_id) + "\"]}";
	sendSocketData(json);

}

void EthStratumClient::submitSolution(Solution solution) {

	string nonceHex = toHex(solution.nonce);
	string json;

	m_responsetimer.cancel();

	switch (m_conn.Version()) {

		case EthStratumClient::STRATUM:

			json = "{\"id\": 4, \"method\": \"mining.submit\", \"params\": [\"" +
				m_conn.User() + "\",\"" + solution.work.job.hex() + "\",\"0x" +
				nonceHex + "\",\"0x" + solution.work.header.hex() + "\",\"0x" +
				solution.mixHash.hex() + "\"]}";
			break;

		case EthStratumClient::ETHPROXY:

			json = "{\"id\": 4, \"worker\":\"" +
				m_worker + "\", \"method\": \"eth_submitWork\", \"params\": [\"0x" +
				nonceHex + "\",\"0x" + solution.work.header.hex() + "\",\"0x" +
				solution.mixHash.hex() + "\"]}";
			break;

		case EthStratumClient::ETHEREUMSTRATUM:

			json = "{\"id\": 4, \"method\": \"mining.submit\", \"params\": [\"" +
				m_conn.User() + "\",\"" + solution.work.job.hex().substr(0, solution.work.job_len) + "\",\"" +
				nonceHex.substr(m_extraNonceHexSize, 16 - m_extraNonceHexSize) + "\"]}";
			break;

	}

	sendSocketData(json);

	m_stale = solution.stale;
	m_response_pending = true;

	m_responsetimer.cancel();
	m_responsetimer.expires_from_now(boost::posix_time::seconds(2));
	m_responsetimer.async_wait(boost::bind(&EthStratumClient::response_timeout_handler, this, boost::asio::placeholders::error));
}

void EthStratumClient::recvSocketData() {
	
	if (m_conn.SecLevel() != SecureLevel::NONE) {

		async_read_until(*m_securesocket, m_recvBuffer, "\n",
			boost::bind(&EthStratumClient::onRecvSocketDataCompleted, this,
				boost::asio::placeholders::error, boost::asio::placeholders::bytes_transferred));
	}
	else {

		async_read_until(*m_nonsecuresocket, m_recvBuffer, "\n",
			boost::bind(&EthStratumClient::onRecvSocketDataCompleted, this,
				boost::asio::placeholders::error, boost::asio::placeholders::bytes_transferred));
	}

}

void EthStratumClient::onRecvSocketDataCompleted(const boost::system::error_code& ec, std::size_t bytes_transferred) {
	
	dev::setThreadName("stratum");

	if (!ec && bytes_transferred > 0) {

		// Extract received message
		std::istream is(&m_recvBuffer);
		std::string message;
		getline(is, message);

		if (!message.empty()) {

			// Test validity of chunk and process
			Json::Value jMsg;
			Json::Reader jRdr;
			if (jRdr.parse(message, jMsg)) {
				processReponse(jMsg);
			}
			else {
				cwarn << "Got invalid Json message :" + jRdr.getFormattedErrorMessages();
			}

		}

		// Eventually keep reading from socket
		if (isConnected()) { recvSocketData(); }

	}
	else
	{
		if (isConnected()) {
			cwarn << "Socket read failed:" << ec.message();
			disconnect();
		}
	}

}

void EthStratumClient::sendSocketData(string const & data) {

	dev::setThreadName("stratum");

	if (!isConnected())
		return;
	
	std::ostream os(&m_sendBuffer);
	os << data << "\n";
	
	if (m_conn.SecLevel() != SecureLevel::NONE) {

		async_write(*m_securesocket, m_sendBuffer,
			boost::bind(&EthStratumClient::onSendSocketDataCompleted, this, boost::asio::placeholders::error));

	}
	else {

		async_write(*m_nonsecuresocket, m_sendBuffer,
			boost::bind(&EthStratumClient::onSendSocketDataCompleted, this, boost::asio::placeholders::error));

	}

}

void EthStratumClient::onSendSocketDataCompleted(const boost::system::error_code& ec) {

	dev::setThreadName("stratum");

	if (ec) {
		if (isConnected()) {
			cwarn << "Socket write failed: " + ec.message();
			disconnect();
		}
	}

}
