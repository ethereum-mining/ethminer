#include <ethminer/buildinfo.h>
#include <libdevcore/Log.h>
#include <ethash/ethash.hpp>

#include "EthStratumClient.h"

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


static void diffToTarget(uint32_t* target, double diff)
{
    uint32_t target2[8];
    uint64_t m;
    int k;

    for (k = 6; k > 0 && diff > 1.0; k--)
        diff /= 4294967296.0;
    m = (uint64_t)(4294901760.0 / diff);
    if (m == 0 && k == 6)
        memset(target2, 0xff, 32);
    else
    {
        memset(target2, 0, 32);
        target2[k] = (uint32_t)m;
        target2[k + 1] = (uint32_t)(m >> 32);
    }

    for (int i = 0; i < 32; i++)
        ((uint8_t*)target)[31 - i] = ((uint8_t*)target2)[i];
}


EthStratumClient::EthStratumClient(int worktimeout, int responsetimeout, bool submitHashrate)
  : PoolClient(),
    m_worktimeout(worktimeout),
    m_responsetimeout(responsetimeout),
    m_io_service(g_io_service),
    m_io_strand(g_io_service),
    m_socket(nullptr),
    m_workloop_timer(g_io_service),
    m_response_plea_times(64),
    m_resolver(g_io_service),
    m_endpoints(),
    m_submit_hashrate(submitHashrate)
{
    if (m_submit_hashrate)
        m_submit_hashrate_id = h256::random().hex();

    // Initialize workloop_timer to infinite wait
    m_workloop_timer.expires_at(boost::posix_time::pos_infin);
    m_workloop_timer.async_wait(m_io_strand.wrap(boost::bind(
        &EthStratumClient::workloop_timer_elapsed, this, boost::asio::placeholders::error)));
    clear_response_pleas();
}

EthStratumClient::~EthStratumClient()
{
    // Do not stop io service.
    // It's global
}

void EthStratumClient::init_socket()
{
    // Prepare Socket
    if (m_conn->SecLevel() != SecureLevel::NONE)
    {
        boost::asio::ssl::context::method method = boost::asio::ssl::context::tls_client;
        if (m_conn->SecLevel() == SecureLevel::TLS12)
            method = boost::asio::ssl::context::tlsv12;

        boost::asio::ssl::context ctx(method);
        m_securesocket = std::make_shared<boost::asio::ssl::stream<boost::asio::ip::tcp::socket>>(
            m_io_service, ctx);
        m_socket = &m_securesocket->next_layer();


        m_securesocket->set_verify_mode(boost::asio::ssl::verify_peer);

#ifdef _WIN32
        HCERTSTORE hStore = CertOpenSystemStore(0, "ROOT");
        if (hStore == nullptr)
        {
            return;
        }

        X509_STORE* store = X509_STORE_new();
        PCCERT_CONTEXT pContext = nullptr;
        while ((pContext = CertEnumCertificatesInStore(hStore, pContext)) != nullptr)
        {
            X509* x509 = d2i_X509(
                nullptr, (const unsigned char**)&pContext->pbCertEncoded, pContext->cbCertEncoded);
            if (x509 != nullptr)
            {
                X509_STORE_add_cert(store, x509);
                X509_free(x509);
            }
        }

        CertFreeCertificateContext(pContext);
        CertCloseStore(hStore, 0);

        SSL_CTX_set_cert_store(ctx.native_handle(), store);
#else
        char* certPath = getenv("SSL_CERT_FILE");
        try
        {
            ctx.load_verify_file(certPath ? certPath : "/etc/ssl/certs/ca-certificates.crt");
        }
        catch (...)
        {
            cwarn << "Failed to load ca certificates. Either the file "
                     "'/etc/ssl/certs/ca-certificates.crt' does not exist";
            cwarn << "or the environment variable SSL_CERT_FILE is set to an invalid or "
                     "inaccessible file.";
            cwarn << "It is possible that certificate verification can fail.";
        }
#endif
    }
    else
    {
        m_nonsecuresocket = std::make_shared<boost::asio::ip::tcp::socket>(m_io_service);
        m_socket = m_nonsecuresocket.get();
    }

    // Activate keep alive to detect disconnects
    unsigned int keepAlive = 10000;

#if defined(_WIN32)
    int32_t timeout = keepAlive;
    setsockopt(
        m_socket->native_handle(), SOL_SOCKET, SO_RCVTIMEO, (const char*)&timeout, sizeof(timeout));
    setsockopt(
        m_socket->native_handle(), SOL_SOCKET, SO_SNDTIMEO, (const char*)&timeout, sizeof(timeout));
#else
    struct timeval tv;
    tv.tv_sec = keepAlive / 1000;
    tv.tv_usec = keepAlive % 1000;
    setsockopt(m_socket->native_handle(), SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    setsockopt(m_socket->native_handle(), SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
#endif
}

void EthStratumClient::connect()
{
    // Prevent unnecessary and potentially dangerous recursion
    if (m_connecting.load(std::memory_order::memory_order_relaxed))
        return;

    // Start timing operations
    m_workloop_timer.expires_from_now(boost::posix_time::milliseconds(m_workloop_interval));
    m_workloop_timer.async_wait(m_io_strand.wrap(boost::bind(
        &EthStratumClient::workloop_timer_elapsed, this, boost::asio::placeholders::error)));


    // Reset status flags
    m_canconnect.store(false, std::memory_order_relaxed);
    m_connected.store(false, std::memory_order_relaxed);
    m_subscribed.store(false, std::memory_order_relaxed);
    m_authorized.store(false, std::memory_order_relaxed);
    m_authpending.store(false, std::memory_order_relaxed);

    // Reset data for ETHEREUMSTRATUM (NiceHash) mode (if previously used)
    // https://github.com/nicehash/Specifications/blob/master/EthereumStratum_NiceHash_v1.0.0.txt
    /*
    "Before first job (work) is provided, pool MUST set difficulty by sending mining.set_difficulty
    If pool does not set difficulty before first job, then miner can assume difficulty 1 was being
    set." Those above statement imply we MAY NOT receive difficulty thus at each new connection
    restart from 1
    */
    m_nextWorkBoundary = h256("0xffff000000000000000000000000000000000000000000000000000000000000");
    m_extraNonce = h64();
    m_extraNonceHexSize = 0;

    // Initializes socket and eventually secure stream
    if (!m_socket)
        init_socket();

    // Begin resolve all ips associated to hostname
    // empty queue from any previous listed ip
    // calling the resolver each time is useful as most
    // load balancer will give Ips in different order
    m_endpoints = std::queue<boost::asio::ip::basic_endpoint<boost::asio::ip::tcp>>();
    m_endpoint = boost::asio::ip::basic_endpoint<boost::asio::ip::tcp>();
    m_resolver = tcp::resolver(m_io_service);
    tcp::resolver::query q(m_conn->Host(), toString(m_conn->Port()));

    // Start resolving async
    m_resolver.async_resolve(
        q, m_io_strand.wrap(boost::bind(&EthStratumClient::resolve_handler, this,
               boost::asio::placeholders::error, boost::asio::placeholders::iterator)));
}

void EthStratumClient::disconnect()
{
    // Prevent unnecessary recursion
    if (!m_connected.load(std::memory_order_relaxed) ||
        m_disconnecting.load(std::memory_order_relaxed))
        return;
    m_disconnecting.store(true, std::memory_order_relaxed);

    // Cancel any outstanding async operation
    if (m_socket)
        m_socket->cancel();

    if (m_socket && m_socket->is_open())
    {
        try
        {
            boost::system::error_code sec;

            if (m_conn->SecLevel() != SecureLevel::NONE)
            {
                // This will initiate the exchange of "close_notify" message among parties.
                // If both client and server are connected then we expect the handler with success
                // As there may be a connection issue we also endorse a timeout
                m_securesocket->async_shutdown(
                    m_io_strand.wrap(boost::bind(&EthStratumClient::onSSLShutdownCompleted, this,
                        boost::asio::placeholders::error)));
                enqueue_response_plea();


                // Rest of disconnection is performed asynchronously
                return;
            }
            else
            {
                m_nonsecuresocket->shutdown(boost::asio::ip::tcp::socket::shutdown_both, sec);
                m_socket->close();
            }
        }
        catch (std::exception const& _e)
        {
            cwarn << "Error while disconnecting:" << _e.what();
        }
    }

    disconnect_finalize();
}

void EthStratumClient::disconnect_finalize()
{
    if (m_conn->SecLevel() != SecureLevel::NONE)
    {
        if (m_securesocket->lowest_layer().is_open())
        {
            // Manage error code if layer is already shut down
            boost::system::error_code ec;
            m_securesocket->lowest_layer().shutdown(
                boost::asio::ip::tcp::socket::shutdown_both, ec);
            m_securesocket->lowest_layer().close();
        }
        m_securesocket = nullptr;
        m_socket = nullptr;
    }
    else
    {
        m_socket = nullptr;
        m_nonsecuresocket = nullptr;
    }

    // Release locking flag and set connection status
#ifdef DEV_BUILD
    if (g_logOptions & LOG_CONNECT)
        cnote << "Socket disconnected from " << ActiveEndPoint();
#endif
    m_connected.store(false, std::memory_order_relaxed);
    m_subscribed.store(false, std::memory_order_relaxed);
    m_authorized.store(false, std::memory_order_relaxed);
    m_authpending.store(false, std::memory_order_relaxed);
    m_disconnecting.store(false, std::memory_order_relaxed);

    if (!m_conn->IsUnrecoverable())
    {
        // If we got disconnected during autodetection phase
        // reissue a connect lowering stratum mode checks
        // m_canconnect flag is used to prevent never-ending loop when
        // remote endpoint rejects connections attempts persistently since the first
        if (!m_conn->StratumModeConfirmed() && m_canconnect.load(std::memory_order_relaxed))
        {
            // Repost a new connection attempt and advance to next stratum test
            if (m_conn->StratumMode() > 0)
            {
                m_conn->SetStratumMode(m_conn->StratumMode() - 1);
                m_io_service.post(
                    m_io_strand.wrap(boost::bind(&EthStratumClient::start_connect, this)));
                return;
            }
            else
            {
                // There are no more stratum modes to test
                // Mark connection as unrecoverable and trash it
                m_conn->MarkUnrecoverable();
            }
        }
    }

    // Clear plea queue and stop timing
    std::chrono::steady_clock::time_point m_response_plea_time;
    clear_response_pleas();
    m_solution_submitted_max_id = 0;

    // Put the actor back to sleep
    m_workloop_timer.expires_at(boost::posix_time::pos_infin);
    m_workloop_timer.async_wait(m_io_strand.wrap(boost::bind(
        &EthStratumClient::workloop_timer_elapsed, this, boost::asio::placeholders::error)));

    // Trigger handlers
    if (m_onDisconnected)
    {
        m_onDisconnected();
    }
}

void EthStratumClient::resolve_handler(
    const boost::system::error_code& ec, tcp::resolver::iterator i)
{
    if (!ec)
    {
        dev::setThreadName("stratum");

        while (i != tcp::resolver::iterator())
        {
            m_endpoints.push(i->endpoint());
            i++;
        }
        m_resolver.cancel();

        // Resolver has finished so invoke connection asynchronously
        m_io_service.post(m_io_strand.wrap(boost::bind(&EthStratumClient::start_connect, this)));
    }
    else
    {
        dev::setThreadName("stratum");
        cwarn << "Could not resolve host " << m_conn->Host() << ", " << ec.message();

        // Release locking flag and set connection status
        m_connected.store(false, std::memory_order_relaxed);
        m_connecting.store(false, std::memory_order_relaxed);

        // Trigger handlers
        if (m_onDisconnected)
        {
            m_onDisconnected();
        }
    }
}

void EthStratumClient::start_connect()
{
    if (m_connecting.load(std::memory_order_relaxed))
        return;
    m_connecting.store(true, std::memory_order::memory_order_relaxed);

    if (!m_endpoints.empty())
    {
        // Pick the first endpoint in list.
        // Eventually endpoints get discarded on connection errors
        m_endpoint = m_endpoints.front();

        // Re-init socket if we need to
        if (m_socket == nullptr)
            init_socket();

        dev::setThreadName("stratum");

#ifdef DEV_BUILD
        if (g_logOptions & LOG_CONNECT)
            cnote << ("Trying " + toString(m_endpoint) + " ...");
#endif

        clear_response_pleas();
        m_connecting.store(true, std::memory_order::memory_order_relaxed);
        enqueue_response_plea();
        m_solution_submitted_max_id = 0;

        // Start connecting async
        if (m_conn->SecLevel() != SecureLevel::NONE)
        {
            m_securesocket->lowest_layer().async_connect(m_endpoint,
                m_io_strand.wrap(boost::bind(&EthStratumClient::connect_handler, this, _1)));
        }
        else
        {
            m_socket->async_connect(m_endpoint,
                m_io_strand.wrap(boost::bind(&EthStratumClient::connect_handler, this, _1)));
        }
    }
    else
    {
        dev::setThreadName("stratum");
        m_connecting.store(false, std::memory_order_relaxed);
        cwarn << "No more IP addresses to try for host: " << m_conn->Host();

        // Trigger handlers
        if (m_onDisconnected)
        {
            m_onDisconnected();
        }
    }
}

void EthStratumClient::workloop_timer_elapsed(const boost::system::error_code& ec)
{
    using namespace std::chrono;

    // On timer cancelled or nothing to check for then early exit
    if (ec == boost::asio::error::operation_aborted)
    {
        return;
    }

    if (m_response_pleas_count.load(std::memory_order_relaxed))
    {
        milliseconds response_delay_ms(0);
        steady_clock::time_point m_response_plea_time(m_response_plea_older.load(std::memory_order_relaxed));

        // Check responses while in connection/disconnection phase
        if (isPendingState())
        {
            response_delay_ms =
                duration_cast<milliseconds>(steady_clock::now() - m_response_plea_time);

            if ((m_responsetimeout * 1000) >= response_delay_ms.count())
            {
                if (m_connecting.load(std::memory_order_relaxed))
                {
                    // The socket is closed so that any outstanding
                    // asynchronous connection operations are cancelled.
                    m_socket->close();
                    return;
                }

                // This is set for SSL disconnection
                if (m_disconnecting.load(std::memory_order_relaxed) &&
                    (m_conn->SecLevel() != SecureLevel::NONE))
                {
                    if (m_securesocket->lowest_layer().is_open())
                    {
                        m_securesocket->lowest_layer().close();
                        return;
                    }
                }
            }
        }

        // Check responses while connected
        if (isConnected())
        {
            response_delay_ms =
                duration_cast<milliseconds>(steady_clock::now() - m_response_plea_time);

            if (response_delay_ms.count() >= (m_responsetimeout * 1000))
            {
                if (m_conn->StratumModeConfirmed() == false && m_conn->IsUnrecoverable() == false)
                {
                    // Waiting for a response from pool to a login request
                    // Async self send a fake error response
                    Json::Value jRes;
                    jRes["id"] = unsigned(1);
                    jRes["result"] = Json::nullValue;
                    jRes["error"] = true;
                    clear_response_pleas();
                    m_io_service.post(m_io_strand.wrap(
                        boost::bind(&EthStratumClient::processResponse, this, jRes)));
                }
                else
                {
                    // Waiting for a response to solution submission
                    dev::setThreadName("stratum");
                    cwarn << "No response received in " << m_responsetimeout << " seconds.";
                    m_endpoints.pop();
                    m_subscribed.store(false, std::memory_order_relaxed);
                    m_authorized.store(false, std::memory_order_relaxed);
                    clear_response_pleas();
                    m_io_service.post(
                        m_io_strand.wrap(boost::bind(&EthStratumClient::disconnect, this)));
                }

            }

            // Check how old is last job received
            if (duration_cast<seconds>(steady_clock::now() - m_current_timestamp).count() >
                m_worktimeout)
            {
                dev::setThreadName("stratum");
                cwarn << "No new work received in " << m_worktimeout << " seconds.";
                m_endpoints.pop();
                m_subscribed.store(false, std::memory_order_relaxed);
                m_authorized.store(false, std::memory_order_relaxed);
                clear_response_pleas();
                m_io_service.post(
                    m_io_strand.wrap(boost::bind(&EthStratumClient::disconnect, this)));
            }
        }
    }

    // Resubmit timing operations
    m_workloop_timer.expires_from_now(boost::posix_time::milliseconds(m_workloop_interval));
    m_workloop_timer.async_wait(m_io_strand.wrap(boost::bind(
        &EthStratumClient::workloop_timer_elapsed, this, boost::asio::placeholders::error)));
}

void EthStratumClient::connect_handler(const boost::system::error_code& ec)
{
    dev::setThreadName("stratum");

    // Set status completion
    m_connecting.store(false, std::memory_order_relaxed);


    // Timeout has run before or we got error
    if (ec || !m_socket->is_open())
    {
        cwarn << ("Error  " + toString(m_endpoint) + " [ " + (ec ? ec.message() : "Timeout") +
                  " ]");

        // We need to close the socket used in the previous connection attempt
        // before starting a new one.
        // In case of error, in fact, boost does not close the socket
        // If socket is not opened it means we got timed out
        if (m_socket->is_open())
            m_socket->close();

        // Discard this endpoint and try the next available.
        // Eventually is start_connect which will check for an
        // empty list.
        m_endpoints.pop();
        m_canconnect.store(false, std::memory_order_relaxed);
        m_io_service.post(m_io_strand.wrap(boost::bind(&EthStratumClient::start_connect, this)));

        return;
    }

    // We got a socket connection established
    m_canconnect.store(true, std::memory_order_relaxed);
#ifdef DEV_BUILD
    if (g_logOptions & LOG_CONNECT)
        cnote << "Socket connected to " << ActiveEndPoint();
#endif

    if (m_conn->SecLevel() != SecureLevel::NONE)
    {
        boost::system::error_code hec;
        m_securesocket->lowest_layer().set_option(boost::asio::socket_base::keep_alive(true));
        m_securesocket->lowest_layer().set_option(tcp::no_delay(true));

        m_securesocket->handshake(boost::asio::ssl::stream_base::client, hec);

        if (hec)
        {
            cwarn << "SSL/TLS Handshake failed: " << hec.message();
            if (hec.value() == 337047686)
            {  // certificate verification failed
                cwarn << "This can have multiple reasons:";
                cwarn << "* Root certs are either not installed or not found";
                cwarn << "* Pool uses a self-signed certificate";
                cwarn << "Possible fixes:";
                cwarn << "* Make sure the file '/etc/ssl/certs/ca-certificates.crt' exists and "
                         "is accessible";
                cwarn << "* Export the correct path via 'export "
                         "SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt' to the correct "
                         "file";
                cwarn << "  On most systems you can install the 'ca-certificates' package";
                cwarn << "  You can also get the latest file here: "
                         "https://curl.haxx.se/docs/caextract.html";
                cwarn << "* Disable certificate verification all-together via command-line "
                         "option.";
            }

            // This is a fatal error
            // No need to try other IPs as the certificate is based on host-name
            // not ip address. Trying other IPs would end up with the very same error.
            m_canconnect.store(false, std::memory_order_relaxed);
            m_conn->MarkUnrecoverable();
            m_io_service.post(m_io_strand.wrap(boost::bind(&EthStratumClient::disconnect, this)));
            return;
        }
    }
    else
    {
        m_nonsecuresocket->set_option(boost::asio::socket_base::keep_alive(true));
        m_nonsecuresocket->set_option(tcp::no_delay(true));
    }

    // Here is where we're properly connected
    m_connected.store(true, std::memory_order_relaxed);

    // Clean buffer from any previous stale data
    m_sendBuffer.consume(4096);
    clear_response_pleas();

    // Extract user and worker
    size_t p;
    m_worker.clear();
    p = m_conn->User().find_first_of(".");
    if (p != string::npos)
    {
        m_user = m_conn->User().substr(0, p);

        // There should be at least one char after dot
        // returned p is zero based
        if (p < (m_conn->User().length() - 1))
            m_worker = m_conn->User().substr(++p);
    }
    else
    {
        m_user = m_conn->User();
    }

    /*

    If connection has been set-up with a specific scheme then
    set it's related stratum version as confirmed.

    Otherwise let's go through an autodetection.

    Autodetection process passes all known stratum modes.
    - 1st pass EthStratumClient::ETHEREUMSTRATUM  (2)
    - 2nd pass EthStratumClient::ETHPROXY         (1)
    - 3rd pass EthStratumClient::STRATUM          (0)
    */

    if (m_conn->Version() < 999)
    {
        m_conn->SetStratumMode(m_conn->Version(), true);
    }
    else
    {
        if (!m_conn->StratumModeConfirmed() && m_conn->StratumMode() == 999)
            m_conn->SetStratumMode(2, false);
    }


    Json::Value jReq;
    jReq["id"] = unsigned(1);
    jReq["method"] = "mining.subscribe";
    jReq["params"] = Json::Value(Json::arrayValue);


    switch (m_conn->StratumMode())
    {
    case EthStratumClient::STRATUM:

        jReq["jsonrpc"] = "2.0";

        break;

    case EthStratumClient::ETHPROXY:

        jReq["method"] = "eth_submitLogin";
        if (m_worker.length())
            jReq["worker"] = m_worker;
        jReq["params"].append(m_user + m_conn->Path());

        break;

    case EthStratumClient::ETHEREUMSTRATUM:

        jReq["params"].append(ethminer_get_buildinfo()->project_name_with_version);
        jReq["params"].append("EthereumStratum/1.0.0");

        break;
    }

    // Begin receive data
    recvSocketData();

    /*
    Send first message
    NOTE !!
    It's been tested that f2pool.com does not respond with json error to wrong
    access message (which is needed to autodetect stratum mode).
    IT DOES NOT RESPOND AT ALL !!
    Due to this we need to set a timeout (arbitrary set to 1 second) and
    if no response within that time consider the tentative login failed
    and switch to next stratum mode test
    */
    enqueue_response_plea();
    sendSocketData(jReq);
}

std::string EthStratumClient::processError(Json::Value& responseObject)
{
    std::string retVar;

    if (responseObject.isMember("error") &&
        !responseObject.get("error", Json::Value::null).isNull())
    {
        if (responseObject["error"].isConvertibleTo(Json::ValueType::stringValue))
        {
            retVar = responseObject.get("error", "Unknown error").asString();
        }
        else if (responseObject["error"].isConvertibleTo(Json::ValueType::arrayValue))
        {
            for (auto i : responseObject["error"])
            {
                retVar += i.asString() + " ";
            }
        }
        else if (responseObject["error"].isConvertibleTo(Json::ValueType::objectValue))
        {
            for (Json::Value::iterator i = responseObject["error"].begin();
                 i != responseObject["error"].end(); ++i)
            {
                Json::Value k = i.key();
                Json::Value v = (*i);
                retVar += (std::string)i.name() + ":" + v.asString() + " ";
            }
        }
    }
    else
    {
        retVar = "Unknown error";
    }

    return retVar;
}

void EthStratumClient::processExtranonce(std::string& enonce)
{
    m_extraNonceHexSize = enonce.length();

    cnote << "Extranonce set to " EthWhite << enonce << EthReset " (nicehash)";
    enonce.append(16 - m_extraNonceHexSize, '0');
    m_extraNonce = h64(enonce);
}

void EthStratumClient::processResponse(Json::Value& responseObject)
{
    dev::setThreadName("stratum");

    // Out received message only for debug purpouses
    if (g_logOptions & LOG_JSON)
        cnote << responseObject;

    // Store jsonrpc version to test against
    int _rpcVer = responseObject.isMember("jsonrpc") ? 2 : 1;

    bool _isNotification = false;  // Whether or not this message is a reply to previous request or
                                   // is a broadcast notification
    bool _isSuccess = false;       // Whether or not this is a succesful or failed response (implies
                                   // _isNotification = false)
    string _errReason = "";        // Content of the error reason
    string _method = "";           // The method of the notification (or request from pool)
    unsigned _id = 0;  // This SHOULD be the same id as the request it is responding to (known exception
                       // is ethermine.org using 999)


    // Retrieve essential values
    _id = responseObject.get("id", unsigned(0)).asUInt();
    _isSuccess = responseObject.get("error", Json::Value::null).empty();
    _errReason = (_isSuccess ? "" : processError(responseObject));
    _method = responseObject.get("method", "").asString();
    _isNotification = (_method != "" || _id == unsigned(0));

    // Notifications of new jobs are like responses to get_work requests
    if (_isNotification && _method == "" && m_conn->StratumMode() == EthStratumClient::ETHPROXY &&
        responseObject["result"].isArray())
    {
        _method = "mining.notify";
    }

    // Very minimal sanity checks
    // - For rpc2 member "jsonrpc" MUST be valued to "2.0"
    // - For responses ... well ... whatever
    // - For notifications I must receive "method" member and a not empty "params" or "result"
    // member
    if ((_rpcVer == 2 && (!responseObject["jsonrpc"].isString() ||
                             responseObject.get("jsonrpc", "") != "2.0")) ||
        (_isNotification && (responseObject["params"].empty() && responseObject["result"].empty())))
    {
        cwarn << "Pool sent an invalid jsonrpc message...";
        cwarn << "Do not blame ethminer for this. Ask pool devs to honor http://www.jsonrpc.org/ "
                 "specifications ";
        cwarn << "Disconnecting...";
        m_subscribed.store(false, std::memory_order_relaxed);
        m_authorized.store(false, std::memory_order_relaxed);
        m_io_service.post(m_io_strand.wrap(boost::bind(&EthStratumClient::disconnect, this)));
        return;
    }


    // Handle awaited responses to OUR requests (calc response times)
    if (!_isNotification)
    {
        Json::Value jReq;
        Json::Value jResult = responseObject.get("result", Json::Value::null);
        std::chrono::milliseconds response_delay_ms(0);

        if (_id == 1)
        {
            response_delay_ms = dequeue_response_plea();

            /*
            This is the response to very first message after connection.
            I wish I could manage to have different Ids but apparently ethermine.org always replies
            to first message with id=1 regardless the id originally sent.
            */

            // If still in detection phase every failure to
            // to our issued method must lead to a disconnection and
            // reconnection with next available method.
            if (!m_conn->StratumModeConfirmed())
            {
                if (!_isSuccess)
                {
                    // Disconnect and Proceed with next step of autodetection
                    m_io_service.post(
                        m_io_strand.wrap(boost::bind(&EthStratumClient::disconnect, this)));
                    return;
                }

                switch (m_conn->StratumMode())
                {
                case EthStratumClient::ETHEREUMSTRATUM:

                    // In case of success we also need to verify third parameter of "result" array
                    // member is exactly "EthereumStratum/1.0.0". Otherwise try with another mode
                    if (jResult.isArray() && jResult[0].isArray() && jResult[0].size() == 3 &&
                        jResult[0].get(Json::Value::ArrayIndex(2), "").asString() ==
                            "EthereumStratum/1.0.0")
                    {
                        // ETHEREUMSTRATUM is confirmed
                        m_conn->SetStratumMode(2, true);
                    }
                    else
                    {
                        // Disconnect and Proceed with next step of autodetection ETHPROXY
                        // compatible
                        m_io_service.post(
                            m_io_strand.wrap(boost::bind(&EthStratumClient::disconnect, this)));
                        return;
                    }

                    break;

                case EthStratumClient::ETHPROXY:

                    // ETHPROXY is confirmed
                    m_conn->SetStratumMode(1, true);

                    break;

                case EthStratumClient::STRATUM:

                    // STRATUM is confirmed
                    m_conn->SetStratumMode(0, true);

                    break;
                }
            }


            // Response to "mining.subscribe"
            // (https://en.bitcoin.it/wiki/Stratum_mining_protocol#mining.subscribe) Result should
            // be an array with multiple dimensions, we only care about the data if
            // EthStratumClient::ETHEREUMSTRATUM
            switch (m_conn->StratumMode())
            {
            case EthStratumClient::STRATUM:

                if (jResult.isArray() && jResult[0].isArray() && jResult[0].size() == 3 &&
                    jResult[0].get(Json::Value::ArrayIndex(2), "").asString() ==
                        "EthereumStratum/1.0.0")
                {
                    _isSuccess = false;
                }
                else
                {
                    cnote << "Stratum mode detected: STRATUM";
                }

                m_subscribed.store(_isSuccess, std::memory_order_relaxed);
                if (!m_subscribed)
                {
                    cnote << "Could not subscribe: " << _errReason;
                    m_conn->MarkUnrecoverable();
                    m_io_service.post(
                        m_io_strand.wrap(boost::bind(&EthStratumClient::disconnect, this)));
                    return;
                }
                else
                {
                    cnote << "Subscribed!";
                    m_authpending.store(true, std::memory_order_relaxed);
                    jReq["id"] = unsigned(3);
                    jReq["jsonrpc"] = "2.0";
                    jReq["method"] = "mining.authorize";
                    jReq["params"] = Json::Value(Json::arrayValue);
                    jReq["params"].append(m_conn->User() + m_conn->Path());
                    jReq["params"].append(m_conn->Pass());
                    enqueue_response_plea();
                }

                break;

            case EthStratumClient::ETHPROXY:

                cnote << "Stratum mode detected: ETHPROXY Compatible";
                m_subscribed.store(_isSuccess, std::memory_order_relaxed);
                if (!m_subscribed)
                {
                    cnote << "Could not login:" << _errReason;
                    m_conn->MarkUnrecoverable();
                    m_io_service.post(
                        m_io_strand.wrap(boost::bind(&EthStratumClient::disconnect, this)));
                    return;
                }
                else
                {
                    cnote << "Logged in!";
                    m_authorized.store(true, std::memory_order_relaxed);

                    // If we get here we have a valid application connection
                    // not only a socket connection
                    if (m_onConnected && m_conn->StratumModeConfirmed())
                    {
                        m_current_timestamp = std::chrono::steady_clock::now();
                        m_onConnected();
                    }

                    jReq["id"] = unsigned(5);
                    jReq["method"] = "eth_getWork";
                    jReq["params"] = Json::Value(Json::arrayValue);
                }

                break;

            case EthStratumClient::ETHEREUMSTRATUM:

                cnote << "Stratum mode detected: ETHEREUMSTRATUM (NiceHash)";
                m_subscribed.store(_isSuccess, std::memory_order_relaxed);
                if (!m_subscribed)
                {
                    cnote << "Could not subscribe: " << _errReason;
                    m_conn->MarkUnrecoverable();
                    m_io_service.post(
                        m_io_strand.wrap(boost::bind(&EthStratumClient::disconnect, this)));
                    return;
                }
                else
                {
                    cnote << "Subscribed to stratum server";

                    if (!jResult.empty() && jResult.isArray())
                    {
                        std::string enonce = jResult.get(Json::Value::ArrayIndex(1), "").asString();
                        processExtranonce(enonce);
                    }

                    // Notify we're ready for extra nonce subscribtion on the fly
                    // reply to this message should not perform any logic
                    jReq["id"] = unsigned(2);
                    jReq["method"] = "mining.extranonce.subscribe";
                    jReq["params"] = Json::Value(Json::arrayValue);
                    sendSocketData(jReq);

                    // Eventually request authorization
                    m_authpending.store(true, std::memory_order_relaxed);
                    jReq["id"] = unsigned(3);
                    jReq["method"] = "mining.authorize";
                    jReq["params"].append(m_conn->User() + m_conn->Path());
                    jReq["params"].append(m_conn->Pass());
                    enqueue_response_plea();
                }


                break;
            }

            sendSocketData(jReq);
        }

        else if (_id == 2)
        {
            // This is the response to mining.extranonce.subscribe
            // according to this
            // https://github.com/nicehash/Specifications/blob/master/NiceHash_extranonce_subscribe_extension.txt
            // In all cases, client does not perform any logic when receiving back these replies.
            // With mining.extranonce.subscribe subscription, client should handle extranonce1
            // changes correctly

            // Nothing to do here.
        }

        else if (_id == 3)
        {
            response_delay_ms = dequeue_response_plea();

            // Response to "mining.authorize"
            // (https://en.bitcoin.it/wiki/Stratum_mining_protocol#mining.authorize) Result should
            // be boolean, some pools also throw an error, so _isSuccess can be false Due to this
            // reevaluate _isSuccess

            if (_isSuccess && jResult.isBool())
            {
                _isSuccess = jResult.asBool();
            }

            m_authpending.store(false, std::memory_order_relaxed);
            m_authorized.store(_isSuccess, std::memory_order_relaxed);

            if (!m_authorized)
            {
                cnote << "Worker not authorized " << m_conn->User() << _errReason;
                m_conn->MarkUnrecoverable();
                m_io_service.post(
                    m_io_strand.wrap(boost::bind(&EthStratumClient::disconnect, this)));
                return;
            }
            else
            {
                cnote << "Authorized worker " + m_conn->User();

                // If we get here we have a valid application connection
                // not only a socket connection
                if (m_onConnected && m_conn->StratumModeConfirmed())
                {
                    m_current_timestamp = std::chrono::steady_clock::now();
                    m_onConnected();
                }
            }
        }

        else if (_id >= 40 && _id <= m_solution_submitted_max_id)
        {
            response_delay_ms = dequeue_response_plea();

            // Response to solution submission mining.submit
            // (https://en.bitcoin.it/wiki/Stratum_mining_protocol#mining.submit) Result should be
            // boolean, some pools also throw an error, so _isSuccess can be false Due to this
            // reevaluate _isSucess

            if (_isSuccess && jResult.isBool())
            {
                _isSuccess = jResult.asBool();
            }

            {
                const unsigned miner_index = _id - 40;
                dequeue_response_plea();
                if (_isSuccess)
                {
                    if (m_onSolutionAccepted)
                    {
                        m_onSolutionAccepted(m_stale, response_delay_ms, miner_index);
                    }
                }
                else
                {
                    if (m_onSolutionRejected)
                    {
                        cwarn << "Reject reason :"
                              << (_errReason.empty() ? "Unspecified" : _errReason);
                        m_onSolutionRejected(m_stale, response_delay_ms, miner_index);
                    }
                }
            }
        }

        else if (_id == 5)
        {
            // This is the response we get on first get_work request issued
            // in mode EthStratumClient::ETHPROXY
            // thus we change it to a mining.notify notification
            if (m_conn->StratumMode() == EthStratumClient::ETHPROXY &&
                responseObject["result"].isArray())
            {
                _method = "mining.notify";
                _isNotification = true;
            }
        }

        else if (_id == 9)
        {
            // Response to hashrate submit
            // Shall we do anything ?
            // Hashrate submit is actually out of stratum spec
            if (!_isSuccess)
            {
                cwarn << "Submit hashRate failed:"
                      << (_errReason.empty() ? "Unspecified error" : _errReason);
            }
        }

        else if (_id == 999)
        {
            // This unfortunate case should not happen as none of the outgoing requests is marked
            // with id 999 However it has been tested that ethermine.org responds with this id when
            // error replying to either mining.subscribe (1) or mining.authorize requests (3) To
            // properly handle this situation we need to rely on Subscribed/Authorized states

            response_delay_ms = dequeue_response_plea();

            if (!_isSuccess)
            {
                if (!m_subscribed)
                {
                    // Subscription pending
                    cnote << "Subscription failed:"
                          << (_errReason.empty() ? "Unspecified error" : _errReason);
                    m_io_service.post(
                        m_io_strand.wrap(boost::bind(&EthStratumClient::disconnect, this)));
                    return;
                }
                else if (m_subscribed && !m_authorized)
                {
                    // Authorization pending
                    cnote << "Worker not authorized:"
                          << (_errReason.empty() ? "Unspecified error" : _errReason);
                    m_io_service.post(
                        m_io_strand.wrap(boost::bind(&EthStratumClient::disconnect, this)));
                    return;
                }
            };
        }

        else
        {

            cnote << "Got response for unknown message id [" << _id << "] Discarding...";
            return;
        }
    }

    /*


    Handle unsolicited messages FROM pool AKA notifications

    NOTE !
    Do not process any notification unless login validated
    which means we have detected proper stratum mode.

    */

    if (_isNotification && m_conn->StratumModeConfirmed())
    {
        Json::Value jReq;
        Json::Value jPrm;

        unsigned prmIdx;

        if (_method == "mining.notify")
        {
            // Discard jobs if not properly subscribed
            if (!m_subscribed.load(std::memory_order_relaxed))
            {
                return;
            }

            /*
            Workaround for Nanopool wrong implementation
            see issue # 1348
            */

            if (m_conn->StratumMode() == EthStratumClient::ETHPROXY &&
                responseObject.isMember("result"))
            {
                jPrm = responseObject.get("result", Json::Value::null);
                prmIdx = 0;
            }
            else
            {
                jPrm = responseObject.get("params", Json::Value::null);
                prmIdx = 1;
            }


            if (jPrm.isArray() && !jPrm.empty())
            {
                string job = jPrm.get(Json::Value::ArrayIndex(0), "").asString();

                if (m_conn->StratumMode() == EthStratumClient::ETHEREUMSTRATUM)
                {
                    string sSeedHash = jPrm.get(Json::Value::ArrayIndex(1), "").asString();
                    string sHeaderHash = jPrm.get(Json::Value::ArrayIndex(2), "").asString();

                    if (sHeaderHash != "" && sSeedHash != "")
                    {
                        m_current.epoch = ethash::find_epoch_number(
                            ethash::hash256_from_bytes(h256{sSeedHash}.data()));
                        m_current.header = h256(sHeaderHash);
                        m_current.boundary = m_nextWorkBoundary;
                        m_current.startNonce = bswap(*((uint64_t*)m_extraNonce.data()));
                        m_current.exSizeBits = m_extraNonceHexSize * 4;
                        m_current.job_len = job.size();
                        job.resize(64, '0');
                        m_current.job = h256(job);
                        m_current_timestamp = std::chrono::steady_clock::now();

                        if (m_onWorkReceived)
                        {
                            m_onWorkReceived(m_current);
                        }
                    }
                }
                else
                {
                    string sHeaderHash = jPrm.get(Json::Value::ArrayIndex(prmIdx++), "").asString();
                    string sSeedHash = jPrm.get(Json::Value::ArrayIndex(prmIdx++), "").asString();
                    string sShareTarget =
                        jPrm.get(Json::Value::ArrayIndex(prmIdx++), "").asString();

                    // coinmine.pl fix
                    int l = sShareTarget.length();
                    if (l < 66)
                        sShareTarget = "0x" + string(66 - l, '0') + sShareTarget.substr(2);


                    if (sHeaderHash != "" && sSeedHash != "" && sShareTarget != "")
                    {
                        m_current.epoch = ethash::find_epoch_number(
                            ethash::hash256_from_bytes(h256{sSeedHash}.data()));
                        m_current.header = h256(sHeaderHash);
                        m_current.boundary = h256(sShareTarget);
                        m_current.job = h256(job);
                        m_current_timestamp = std::chrono::steady_clock::now();

                        if (m_onWorkReceived)
                        {
                            m_onWorkReceived(m_current);
                        }
                    }
                }
            }
        }
        else if (_method == "mining.set_difficulty")
        {
            if (m_conn->StratumMode() == EthStratumClient::ETHEREUMSTRATUM)
            {
                jPrm = responseObject.get("params", Json::Value::null);
                if (jPrm.isArray())
                {
                    double nextWorkDifficulty =
                        max(jPrm.get(Json::Value::ArrayIndex(0), 1).asDouble(), 0.0001);
                    diffToTarget((uint32_t*)m_nextWorkBoundary.data(), nextWorkDifficulty);
#ifdef DEV_BUILD
                    if (g_logOptions & LOG_CONNECT)
                        cnote << "Difficulty set to " EthWhite << nextWorkDifficulty
                              << EthReset " (nicehash)";
#endif
                }
            }
            else
            {
                cwarn << "Invalid mining.set_difficulty rpc method. Disconnecting ...";
                if (m_conn->StratumModeConfirmed())
                {
                    m_conn->MarkUnrecoverable();
                }
                m_io_service.post(
                    m_io_strand.wrap(boost::bind(&EthStratumClient::disconnect, this)));
            }
        }
        else if (_method == "mining.set_extranonce" &&
                 m_conn->StratumMode() == EthStratumClient::ETHEREUMSTRATUM)
        {
            jPrm = responseObject.get("params", Json::Value::null);
            if (jPrm.isArray())
            {
                std::string enonce = jPrm.get(Json::Value::ArrayIndex(0), "").asString();
                if (!enonce.empty())
                    processExtranonce(enonce);
            }
        }
        else if (_method == "client.get_version")
        {
            jReq["id"] = toString(_id);
            jReq["result"] = ethminer_get_buildinfo()->project_name_with_version;

            if (_rpcVer == 1)
            {
                jReq["error"] = Json::Value::null;
            }
            else if (_rpcVer == 2)
            {
                jReq["jsonrpc"] = "2.0";
            }

            sendSocketData(jReq);
        }
        else
        {
            cwarn << "Got unknown method [" << _method << "] from pool. Discarding...";

            // Respond back to issuer
            if (_rpcVer == 2)
            {
                jReq["jsonrpc"] = "2.0";
            }
            jReq["id"] = toString(_id);
            jReq["error"] = "Method not found";

            sendSocketData(jReq);
        }
    }
}

void EthStratumClient::submitHashrate(string const& rate)
{
    m_rate = rate;

    if (!m_submit_hashrate || !isConnected())
    {
        return;
    }

    // There is no stratum method to submit the hashrate so we use the rpc variant.
    // Note !!
    // id = 6 is also the id used by ethermine.org and nanopool to push new jobs
    // thus we will be in trouble if we want to check the result of hashrate submission
    // actually change the id from 6 to 9

    Json::Value jReq;
    jReq["id"] = unsigned(9);
    jReq["jsonrpc"] = "2.0";
    if (m_worker.length())
        jReq["worker"] = m_worker;
    jReq["method"] = "eth_submitHashrate";
    jReq["params"] = Json::Value(Json::arrayValue);
    jReq["params"].append(m_rate);
    jReq["params"].append("0x" + toString(this->m_submit_hashrate_id));

    sendSocketData(jReq);
}

void EthStratumClient::submitSolution(const Solution& solution)
{
    if (!m_subscribed.load(std::memory_order_relaxed) ||
        !m_authorized.load(std::memory_order_relaxed))
    {
        cwarn << "Not authorized";
        return;
    }

    string nonceHex = toHex(solution.nonce);

    Json::Value jReq;

    unsigned id = 40 + solution.index;
    jReq["id"] = id;
    m_solution_submitted_max_id = max(m_solution_submitted_max_id, id);
    jReq["method"] = "mining.submit";
    jReq["params"] = Json::Value(Json::arrayValue);

    switch (m_conn->StratumMode())
    {
    case EthStratumClient::STRATUM:

        jReq["jsonrpc"] = "2.0";
        jReq["params"].append(m_conn->User());
        jReq["params"].append(solution.work.job.hex());
        jReq["params"].append("0x" + nonceHex);
        jReq["params"].append("0x" + solution.work.header.hex());
        jReq["params"].append("0x" + solution.mixHash.hex());
        if (m_worker.length())
            jReq["worker"] = m_worker;

        break;

    case EthStratumClient::ETHPROXY:

        jReq["method"] = "eth_submitWork";
        jReq["params"].append("0x" + nonceHex);
        jReq["params"].append("0x" + solution.work.header.hex());
        jReq["params"].append("0x" + solution.mixHash.hex());
        if (m_worker.length())
            jReq["worker"] = m_worker;

        break;

    case EthStratumClient::ETHEREUMSTRATUM:

        jReq["params"].append(m_conn->User());
        jReq["params"].append(solution.work.job.hex().substr(0, solution.work.job_len));
        jReq["params"].append(nonceHex.substr(m_extraNonceHexSize, 16 - m_extraNonceHexSize));

        break;
    }

    enqueue_response_plea();
    sendSocketData(jReq);

    m_stale = solution.stale;
}

void EthStratumClient::recvSocketData()
{
    if (m_conn->SecLevel() != SecureLevel::NONE)
    {
        async_read_until(*m_securesocket, m_recvBuffer, "\n",
            m_io_strand.wrap(boost::bind(&EthStratumClient::onRecvSocketDataCompleted, this,
                boost::asio::placeholders::error, boost::asio::placeholders::bytes_transferred)));
    }
    else
    {
        async_read_until(*m_nonsecuresocket, m_recvBuffer, "\n",
            m_io_strand.wrap(boost::bind(&EthStratumClient::onRecvSocketDataCompleted, this,
                boost::asio::placeholders::error, boost::asio::placeholders::bytes_transferred)));
    }
}

void EthStratumClient::onRecvSocketDataCompleted(
    const boost::system::error_code& ec, std::size_t bytes_transferred)
{
    dev::setThreadName("stratum");

    // Due to the nature of io_service's queue and
    // the implementation of the loop this event may trigger
    // late after clean disconnection. Check status of connection
    // before triggering all stack of calls

    if (!ec && bytes_transferred > 0)
    {
        // Extract received message
        std::istream is(&m_recvBuffer);
        std::string message;
        getline(is, message);

        if (isConnected())
        {
            if (!message.empty())
            {
                // Test validity of chunk and process
                Json::Value jMsg;
                Json::Reader jRdr;
                if (jRdr.parse(message, jMsg))
                {
                    m_io_service.post(boost::bind(&EthStratumClient::processResponse, this, jMsg));
                }
                else
                {
                    cwarn << "Got invalid Json message :" + jRdr.getFormattedErrorMessages();
                }
            }

            // Eventually keep reading from socket
            recvSocketData();
        }
    }
    else
    {
        if (isConnected())
        {
            if (m_authpending.load(std::memory_order_relaxed))
            {
                cwarn << "Error while waiting for authorization from pool";
                cwarn << "Double check your pool credentials.";
                m_conn->MarkUnrecoverable();
            }

            if ((ec.category() == boost::asio::error::get_ssl_category()) &&
                (ERR_GET_REASON(ec.value()) == SSL_RECEIVED_SHUTDOWN))
            {
                cnote << "SSL Stream remotely closed by " << m_conn->Host();
            }
            else if (ec == boost::asio::error::eof)
            {
                cnote << "Connection remotely closed by " << m_conn->Host();
            }
            else
            {
                cwarn << "Socket read failed: " << ec.message();
            }
            m_io_service.post(m_io_strand.wrap(boost::bind(&EthStratumClient::disconnect, this)));
        }
    }
}

void EthStratumClient::sendSocketData(Json::Value const& jReq)
{
    if (!isConnected())
        return;

    // Out received message only for debug purpouses
    if (g_logOptions & LOG_JSON)
    {
        cnote << jReq;
    }

    std::ostream os(&m_sendBuffer);
    os << m_jWriter.write(jReq);  // Do not add lf. It's added by writer.

    if (m_conn->SecLevel() != SecureLevel::NONE)
    {
        async_write(*m_securesocket, m_sendBuffer,
            m_io_strand.wrap(boost::bind(&EthStratumClient::onSendSocketDataCompleted, this,
                boost::asio::placeholders::error)));
    }
    else
    {
        async_write(*m_nonsecuresocket, m_sendBuffer,
            m_io_strand.wrap(boost::bind(&EthStratumClient::onSendSocketDataCompleted, this,
                boost::asio::placeholders::error)));
    }
}

void EthStratumClient::onSendSocketDataCompleted(const boost::system::error_code& ec)
{
    if (ec)
    {
        if ((ec.category() == boost::asio::error::get_ssl_category()) &&
            (SSL_R_PROTOCOL_IS_SHUTDOWN == ERR_GET_REASON(ec.value())))
        {
            cnote << "SSL Stream error :" << ec.message();
            m_io_strand.wrap(boost::bind(&EthStratumClient::disconnect, this));
        }

        if (isConnected())
        {
            dev::setThreadName("stratum");
            cwarn << "Socket write failed: " + ec.message();
            m_io_strand.wrap(boost::bind(&EthStratumClient::disconnect, this));
        }
    }
}

void EthStratumClient::onSSLShutdownCompleted(const boost::system::error_code& ec)
{
    (void)ec;
    // cnote << "onSSLShutdownCompleted Error code is: " << ec.message();
    m_io_service.post(m_io_strand.wrap(boost::bind(&EthStratumClient::disconnect_finalize, this)));
}

void EthStratumClient::enqueue_response_plea()
{
    using namespace std::chrono;
    steady_clock::time_point response_plea_time = steady_clock::now();
    if (m_response_pleas_count++ == 0)
    {
        m_response_plea_older.store(
            response_plea_time.time_since_epoch(), std::memory_order_relaxed);
    }
    m_response_plea_times.push(response_plea_time);
}

std::chrono::milliseconds EthStratumClient::dequeue_response_plea()
{
    using namespace std::chrono;

    steady_clock::time_point response_plea_time(m_response_plea_older.load(std::memory_order_relaxed));
    milliseconds response_delay_ms =
        duration_cast<milliseconds>(steady_clock::now() - response_plea_time);

    if (m_response_plea_times.pop(response_plea_time))
    {
        m_response_plea_older.store(
            response_plea_time.time_since_epoch(), std::memory_order_relaxed);
    }
    if (m_response_pleas_count.load(std::memory_order_relaxed) > 0)
    {
        m_response_pleas_count--;
        return response_delay_ms;
    }
    else
    {
        return milliseconds(0);
    }
}

void EthStratumClient::clear_response_pleas()
{
    using namespace std::chrono;
    steady_clock::time_point response_plea_time;
    m_response_pleas_count.store(0, std::memory_order_relaxed);
    while (m_response_plea_times.pop(response_plea_time))
    {
    };
    m_response_plea_older.store(((steady_clock::time_point)steady_clock::now()).time_since_epoch(),
        std::memory_order_relaxed);
}
