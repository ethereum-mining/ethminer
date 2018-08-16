#include "ApiServer.h"

#include <ethminer/buildinfo.h>


/* helper functions getting values from a JSON request */
static bool getRequestValue(const char* membername, bool& refValue, Json::Value& jRequest,
    bool optional, Json::Value& jResponse)
{
    if (!jRequest.isMember(membername))
    {
        if (!optional)
        {
            jResponse["error"]["code"] = -32602;
            jResponse["error"]["message"] =
                std::string("Missing '") + std::string(membername) + std::string("'");
        }
        return optional;
    }
    if (!jRequest[membername].isBool())
    {
        jResponse["error"]["code"] = -32602;
        jResponse["error"]["message"] =
            std::string("Invalid type of value '") + std::string(membername) + std::string("'");
        return false;
    }
    if (jRequest[membername].empty())
    {
        jResponse["error"]["code"] = -32602;
        jResponse["error"]["message"] =
            std::string("Empty '") + std::string(membername) + std::string("'");
        return false;
    }
    refValue = jRequest[membername].asBool();
    return true;
}

static bool getRequestValue(const char* membername, unsigned& refValue, Json::Value& jRequest,
    bool optional, Json::Value& jResponse)
{
    if (!jRequest.isMember(membername))
    {
        if (!optional)
        {
            jResponse["error"]["code"] = -32602;
            jResponse["error"]["message"] =
                std::string("Missing '") + std::string(membername) + std::string("'");
        }
        return optional;
    }
    if (!jRequest[membername].isUInt())
    {
        jResponse["error"]["code"] = -32602;
        jResponse["error"]["message"] =
            std::string("Invalid type of value '") + std::string(membername) + std::string("'");
        return false;
    }
    if (jRequest[membername].empty())
    {
        jResponse["error"]["code"] = -32602;
        jResponse["error"]["message"] =
            std::string("Empty '") + std::string(membername) + std::string("'");
        return false;
    }
    refValue = jRequest[membername].asUInt();
    return true;
}

static bool getRequestValue(const char* membername, uint64_t& refValue, Json::Value& jRequest,
    bool optional, Json::Value& jResponse)
{
    if (!jRequest.isMember(membername))
    {
        if (!optional)
        {
            jResponse["error"]["code"] = -32602;
            jResponse["error"]["message"] =
                std::string("Missing '") + std::string(membername) + std::string("'");
        }
        return optional;
    }
    /* as there is no isUInt64() function we can not check the type */
    if (jRequest[membername].empty())
    {
        jResponse["error"]["code"] = -32602;
        jResponse["error"]["message"] =
            std::string("Empty '") + std::string(membername) + std::string("'");
        return false;
    }
    try
    {
        refValue = jRequest[membername].asUInt64();
    }
    catch (...)
    {
        jRequest["error"]["code"] = -32602;
        jResponse["error"]["message"] =
            std::string("Bad value in '") + std::string(membername) + std::string("'");
        return false;
    }
    return true;
}

static bool getRequestValue(const char* membername, Json::Value& refValue, Json::Value& jRequest,
    bool optional, Json::Value& jResponse)
{
    if (!jRequest.isMember(membername))
    {
        if (!optional)
        {
            jResponse["error"]["code"] = -32602;
            jResponse["error"]["message"] =
                std::string("Missing '") + std::string(membername) + std::string("'");
        }
        return optional;
    }
    if (!jRequest[membername].isObject())
    {
        jResponse["error"]["code"] = -32602;
        jResponse["error"]["message"] =
            std::string("Invalid type of value '") + std::string(membername) + std::string("'");
        return false;
    }
    if (jRequest[membername].empty())
    {
        jResponse["error"]["code"] = -32602;
        jResponse["error"]["message"] =
            std::string("Empty '") + std::string(membername) + std::string("'");
        return false;
    }
    refValue = jRequest[membername];
    return true;
}

static bool getRequestValue(const char* membername, std::string& refValue, Json::Value& jRequest,
    bool optional, Json::Value& jResponse)
{
    if (!jRequest.isMember(membername))
    {
        if (!optional)
        {
            jResponse["error"]["code"] = -32602;
            jResponse["error"]["message"] =
                std::string("Missing '") + std::string(membername) + std::string("'");
        }
        return optional;
    }
    if (!jRequest[membername].isString())
    {
        jResponse["error"]["code"] = -32602;
        jResponse["error"]["message"] =
            std::string("Invalid type of value '") + std::string(membername) + std::string("'");
        return false;
    }
    if (jRequest[membername].empty())
    {
        jResponse["error"]["code"] = -32602;
        jResponse["error"]["message"] =
            std::string("Empty '") + std::string(membername) + std::string("'");
        return false;
    }
    refValue = jRequest[membername].asString();
    return true;
}

static bool checkApiWriteAccess(bool is_read_only, Json::Value& jResponse)
{
    if (is_read_only)
    {
        jResponse["error"]["code"] = -32601;
        jResponse["error"]["message"] = "Method not available";
    }
    return !is_read_only;
}

static bool parseRequestId(Json::Value& jRequest, Json::Value& jResponse)
{
    const char* membername = "id";

    // NOTE: all errors have the same code (-32600) indicating this is an invalid request

    // be sure id is there and it's not empty, otherwise raise an error
    if (!jRequest.isMember(membername) || jRequest[membername].empty())
    {
        jResponse[membername] = Json::nullValue;
        jResponse["error"]["code"] = -32600;
        jResponse["error"]["message"] = "Invalid Request (missing or empty id)";
        return false;
    }

    // try to parse id as Uint
    if (jRequest[membername].isUInt())
    {
        jResponse[membername] = jRequest[membername].asUInt();
        return true;
    }

    // try to parse id as String
    if (jRequest[membername].isString())
    {
        jResponse[membername] = jRequest[membername].asString();
        return true;
    }

    // id has invalid type
    jResponse[membername] = Json::nullValue;
    jResponse["error"]["code"] = -32600;
    jResponse["error"]["message"] = "Invalid Request (id has invalid type)";
    return false;
}

ApiServer::ApiServer(boost::asio::io_service& io_service, string address, int portnum,
    bool readonly, string password, Farm& f, PoolManager& mgr)
  : m_readonly(readonly),
    m_password(std::move(password)),
    m_address(address),
    m_portnumber(portnum),
    m_acceptor(io_service),
    m_io_strand(io_service),
    m_farm(f),
    m_mgr(mgr)
{}

void ApiServer::start()
{
    // cnote << "ApiServer::start";
    if (m_portnumber == 0)
        return;

    m_running.store(true, std::memory_order_relaxed);

    tcp::endpoint endpoint(boost::asio::ip::address::from_string(m_address), m_portnumber);

    // Try to bind to port number
    // if exception occurs it may be due to the fact that
    // requested port is already in use by another service
    try
    {
        m_acceptor.open(endpoint.protocol());
        m_acceptor.set_option(boost::asio::ip::tcp::acceptor::reuse_address(true));
        m_acceptor.bind(endpoint);
        m_acceptor.listen(64);
    }
    catch (const std::exception&)
    {
        cwarn << "Could not start API server on port : " +
                     to_string(m_acceptor.local_endpoint().port());
        cwarn << "Ensure port is not in use by another service";
        return;
    }

    cnote << "Api server listening on port " + to_string(m_acceptor.local_endpoint().port())
          << (m_password.empty() ? "." : ". Authentication needed.");
    m_workThread = std::thread{boost::bind(&ApiServer::begin_accept, this)};
}

void ApiServer::stop()
{
    // Exit if not started
    if (!m_running.load(std::memory_order_relaxed))
        return;

    m_acceptor.cancel();
    m_acceptor.close();
    m_running.store(false, std::memory_order_relaxed);

    // Dispose all sessions (if any)
    m_sessions.clear();
}

void ApiServer::begin_accept()
{
    if (!isRunning())
        return;

    dev::setThreadName("Api");
    std::shared_ptr<ApiConnection> session = std::make_shared<ApiConnection>(
        m_acceptor.get_io_service(), ++lastSessionId, m_readonly, m_password, m_farm, m_mgr);
    m_acceptor.async_accept(
        session->socket(), m_io_strand.wrap(boost::bind(&ApiServer::handle_accept, this, session,
                               boost::asio::placeholders::error)));
}

void ApiServer::handle_accept(std::shared_ptr<ApiConnection> session, boost::system::error_code ec)
{
    // Start new connection
    // cnote << "ApiServer::handle_accept";
    if (!ec)
    {
        session->onDisconnected([&](int id) {
            // Destroy pointer to session
            auto it = find_if(m_sessions.begin(), m_sessions.end(),
                [&id](const std::shared_ptr<ApiConnection> session) {
                    return session->getId() == id;
                });
            if (it != m_sessions.end())
            {
                auto index = std::distance(m_sessions.begin(), it);
                m_sessions.erase(m_sessions.begin() + index);
            }
        });
        dev::setThreadName("Api");
        m_sessions.push_back(session);
        cnote << "New api session from " << session->socket().remote_endpoint();
        session->start();
    }
    else
    {
        session.reset();
    }

    // Resubmit new accept
    begin_accept();
}

void ApiConnection::disconnect()
{
    // cnote << "ApiConnection::disconnect";

    // Cancel pending operations
    m_socket.cancel();

    if (m_socket.is_open())
    {
        boost::system::error_code ec;
        m_socket.shutdown(boost::asio::ip::tcp::socket::shutdown_both, ec);
        m_socket.close(ec);
    }

    if (m_onDisconnected)
    {
        m_onDisconnected(this->getId());
    }
}

void ApiConnection::start()
{
    // cnote << "ApiConnection::start";
    recvSocketData();
}

void ApiConnection::processRequest(Json::Value& jRequest, Json::Value& jResponse)
{
    jResponse["jsonrpc"] = "2.0";

    // Strict sanity checks over jsonrpc v2
    if (!parseRequestId(jRequest, jResponse))
        return;

    std::string jsonrpc;
    std::string _method;
    if (!getRequestValue("jsonrpc", jsonrpc, jRequest, false, jResponse) || jsonrpc != "2.0" ||
        !getRequestValue("method", _method, jRequest, false, jResponse))
    {
        jResponse["error"]["code"] = -32600;
        jResponse["error"]["message"] = "Invalid Request";
        return;
    }

    // Check authentication
    if (!m_is_authenticated || _method == "api_authorize")
    {
        if (_method != "api_authorize")
        {
            // Use error code like http 403 Forbidden
            jResponse["error"]["code"] = -403;
            jResponse["error"]["message"] = "Authorization needed";
            return;
        }

        m_is_authenticated =
            false; /* we allow api_authorize method even if already authenticated */

        Json::Value jRequestParams;
        if (!getRequestValue("params", jRequestParams, jRequest, false, jResponse))
            return;

        std::string psw;
        if (!getRequestValue("psw", psw, jRequestParams, false, jResponse))
            return;

        // max password length that we actually verify
        // (this limit can be removed by introducing a collision-resistant compressing hash,
        //  like blake2b/sha3, but 500 should suffice and is much easier to implement)
        const int max_length = 500;
        char input_copy[max_length] = {0};
        char password_copy[max_length] = {0};
        // note: copy() is not O(1) , but i don't think it matters
        psw.copy(&input_copy[0], max_length);
        // ps, the following line can be optimized to only run once on startup and thus save a
        // minuscule amount of cpu cycles.
        m_password.copy(&password_copy[0], max_length);
        int result = 0;
        for (int i = 0; i < max_length; ++i)
        {
            result |= input_copy[i] ^ password_copy[i];
        }

        if (result == 0)
        {
            m_is_authenticated = true;
        }
        else
        {
            // Use error code like http 401 Unauthorized
            jResponse["error"]["code"] = -401;
            jResponse["error"]["message"] = "Invalid password";
            cerr << "Invalid API password provided.";
            // Should we close the connection in the outer function after invalid password ?
        }
        /*
         * possible wait here a fixed time of eg 10s before respond after 5 invalid
           authentications were submitted to prevent brute force password attacks.
        */
        return;
    }

    assert(m_is_authenticated);

    if (_method == "miner_getstat1")
    {
        jResponse["result"] = getMinerStat1();
    }

    else if (_method == "miner_getstathr")
    {
        jResponse["result"] = getMinerStatHR();
    }

    else if (_method == "miner_shuffle")
    {
        // Gives nonce scrambler a new range
        cnote << "Miner Shuffle requested";
        jResponse["result"] = true;
        m_farm.shuffle();
    }

    else if (_method == "miner_ping")
    {
        // Replies back to (check for liveness)
        jResponse["result"] = "pong";
    }

    else if (_method == "miner_restart")
    {
        // Send response to client of success
        // and invoke an async restart
        // to prevent locking
        if (!checkApiWriteAccess(m_readonly, jResponse))
            return;
        cnote << "Miner Restart requested";
        jResponse["result"] = true;
        m_farm.restart_async();
    }

    else if (_method == "miner_reboot")
    {
        if (!checkApiWriteAccess(m_readonly, jResponse))
            return;
        cnote << "Miner reboot requested";

        jResponse["result"] = m_farm.reboot({{"api_miner_reboot"}});
    }

    else if (_method == "miner_getconnections")
    {
        // Returns a list of configured pools
        jResponse["result"] = m_mgr.getConnectionsJson();
    }

    else if (_method == "miner_addconnection")
    {
        if (!checkApiWriteAccess(m_readonly, jResponse))
            return;

        Json::Value jRequestParams;
        if (!getRequestValue("params", jRequestParams, jRequest, false, jResponse))
            return;

        std::string sUri;
        if (!getRequestValue("uri", sUri, jRequestParams, false, jResponse))
            return;

        try
        {
            URI uri(sUri);
            if (!uri.Valid())
            {
                jResponse["error"]["code"] = -422;
                jResponse["error"]["message"] = ("Invalid URI " + uri.String());
                return;
            }
            if (!uri.KnownScheme())
            {
                jResponse["error"]["code"] = -422;
                jResponse["error"]["message"] = ("Unknown URI scheme " + uri.Scheme());
                return;
            }

            // Check other pools already present share the same scheme family (stratum or getwork)
            Json::Value pools = m_mgr.getConnectionsJson();
            for (Json::Value::ArrayIndex i = 0; i != pools.size(); i++)
            {
                dev::URI poolUri = pools[i]["uri"].asString();
                if (uri.Family() != poolUri.Family())
                {
                    jResponse["error"]["code"] = -422;
                    jResponse["error"]["message"] =
                        "Mixed stratum and getwork endpoints not supported.";
                    return;
                }
            }

            // If everything ok then add this new uri
            m_mgr.addConnection(uri);
            jResponse["result"] = true;
        }
        catch (...)
        {
            jResponse["error"]["code"] = -422;
            jResponse["error"]["message"] = "Bad URI";
        }
    }

    else if (_method == "miner_setactiveconnection")
    {
        if (!checkApiWriteAccess(m_readonly, jResponse))
            return;

        Json::Value jRequestParams;
        if (!getRequestValue("params", jRequestParams, jRequest, false, jResponse))
            return;

        unsigned index;
        if (!getRequestValue("index", index, jRequestParams, false, jResponse))
            return;

        Json::Value pools = m_mgr.getConnectionsJson();
        if (index >= pools.size())
        {
            jResponse["error"]["code"] = -422;
            jResponse["error"]["message"] = "Index out of bounds";
            return;
        }

        m_mgr.setActiveConnection(index);
        jResponse["result"] = true;
    }

    else if (_method == "miner_removeconnection")
    {
        if (!checkApiWriteAccess(m_readonly, jResponse))
            return;

        Json::Value jRequestParams;
        if (!getRequestValue("params", jRequestParams, jRequest, false, jResponse))
            return;

        unsigned index;
        if (!getRequestValue("index", index, jRequestParams, false, jResponse))
            return;

        Json::Value pools = m_mgr.getConnectionsJson();
        if (index >= pools.size())
        {
            jResponse["error"]["code"] = -422;
            jResponse["error"]["message"] = "Index out of bounds";
            return;
        }
        if (pools[index]["active"].asBool())
        {
            jResponse["error"]["code"] = -460;
            jResponse["error"]["message"] = "Can't delete active connection";
            return;
        }

        m_mgr.removeConnection(index);
        jResponse["result"] = true;
    }

    else if (_method == "miner_getscramblerinfo")
    {
        jResponse["result"] = m_farm.get_nonce_scrambler_json();
    }

    else if (_method == "miner_setscramblerinfo")
    {
        if (!checkApiWriteAccess(m_readonly, jResponse))
            return;

        Json::Value jRequestParams;
        if (!getRequestValue("params", jRequestParams, jRequest, false, jResponse))
            return;

        bool any_value_provided = false;
        uint64_t nonce = m_farm.get_nonce_scrambler();
        unsigned exp = m_farm.get_segment_width();

        if (!getRequestValue("noncescrambler", nonce, jRequestParams, true, jResponse))
            return;
        any_value_provided = true;

        if (!getRequestValue("segmentwidth", exp, jRequestParams, true, jResponse))
            return;
        any_value_provided = true;

        if (!any_value_provided)
        {
            jResponse["error"]["code"] = -32602;
            jResponse["error"]["message"] = "Missing parameters";
            return;
        }

        if (exp < 10)
            exp = 10;  // Not below
        if (exp > 50)
            exp = 40;  // Not above
        m_farm.set_nonce_scrambler(nonce);
        m_farm.set_nonce_segment_width(exp);
        jResponse["result"] = true;
    }

    else if (_method == "miner_pausegpu")
    {
        if (!checkApiWriteAccess(m_readonly, jResponse))
            return;

        Json::Value jRequestParams;
        if (!getRequestValue("params", jRequestParams, jRequest, false, jResponse))
            return;

        unsigned index;
        if (!getRequestValue("index", index, jRequestParams, false, jResponse))
            return;

        bool pause;
        if (!getRequestValue("pause", pause, jRequestParams, false, jResponse))
            return;

        WorkingProgress p = m_farm.miningProgress(false, false);
        if (index >= p.miningIsPaused.size())
        {
            jResponse["error"]["code"] = -422;
            jResponse["error"]["message"] = "Index out of bounds";
            return;
        }

        auto miner = m_farm.getMiner(index);
        if (pause)
        {
            miner->set_mining_paused(MinigPauseReason::MINING_PAUSED_API);
        }
        else
        {
            miner->clear_mining_paused(MinigPauseReason::MINING_PAUSED_API);
        }
        jResponse["result"] = true;
    }

    else if (_method == "miner_setverbosity")
    {
        if (!checkApiWriteAccess(m_readonly, jResponse))
            return;

        Json::Value jRequestParams;
        if (!getRequestValue("params", jRequestParams, jRequest, false, jResponse))
            return;

        unsigned verbosity;
        if (!getRequestValue("verbosity", verbosity, jRequestParams, false, jResponse))
            return;

        if (verbosity > 9)
        {
            jResponse["error"]["code"] = -422;
            jResponse["error"]["message"] = "Verbosity out of bounds (0-9)";
            return;
        }
        cnote << "Setting verbosity level to " << verbosity;
        g_logVerbosity = verbosity;
        jResponse["result"] = true;
    }

    else
    {
        // Any other method not found
        jResponse["error"]["code"] = -32601;
        jResponse["error"]["message"] = "Method not found";
    }
}

void ApiConnection::recvSocketData()
{
    // cnote << "ApiConnection::recvSocketData";
    boost::asio::async_read_until(m_socket, m_recvBuffer, "\n",
        m_io_strand.wrap(boost::bind(&ApiConnection::onRecvSocketDataCompleted, this,
            boost::asio::placeholders::error, boost::asio::placeholders::bytes_transferred)));
}

void ApiConnection::onRecvSocketDataCompleted(
    const boost::system::error_code& ec, std::size_t bytes_transferred)
{
    // cnote << "ApiConnection::onRecvSocketDataCompleted";
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

        if (m_socket.is_open())
        {
            if (!message.empty())
            {
                // Test validity of chunk and process
                Json::Value jMsg;
                Json::Value jRes;
                Json::Reader jRdr;
                if (jRdr.parse(message, jMsg))
                {
                    processRequest(jMsg, jRes);
                }
                else
                {
                    jRes["jsonrpc"] = "2.0";
                    jRes["id"] = Json::nullValue;
                    jRes["error"]["code"] = -32700;
                    jRes["error"]["message"] = "Parse Error";
                }
                sendSocketData(jRes);
            }

            // Eventually keep reading from socket
            recvSocketData();
        }
    }
    else
    {
        if (m_socket.is_open())
        {
            disconnect();
        }
    }
}

void ApiConnection::sendSocketData(Json::Value const& jReq)
{
    if (!m_socket.is_open())
        return;

    std::ostream os(&m_sendBuffer);
    os << m_jWriter.write(jReq);  // Do not add lf. It's added by writer.

    async_write(m_socket, m_sendBuffer,
        m_io_strand.wrap(boost::bind(
            &ApiConnection::onSendSocketDataCompleted, this, boost::asio::placeholders::error)));
}

void ApiConnection::onSendSocketDataCompleted(const boost::system::error_code& ec)
{
    if (ec)
        disconnect();
}

Json::Value ApiConnection::getMinerStat1()
{
    auto runningTime = std::chrono::duration_cast<std::chrono::minutes>(
        steady_clock::now() - this->m_farm.farmLaunched());

    SolutionStats s = m_farm.getSolutionStats();
    WorkingProgress p = m_farm.miningProgress(true);

    ostringstream totalMhEth;
    ostringstream totalMhDcr;
    ostringstream detailedMhEth;
    ostringstream detailedMhDcr;
    ostringstream tempAndFans;
    ostringstream poolAddresses;
    ostringstream invalidStats;

    totalMhEth << std::fixed << std::setprecision(0) << (p.rate() / 1000.0f) << ";"
               << s.getAccepts() << ";" << s.getRejects();
    totalMhDcr << "0;0;0";                    // DualMining not supported
    invalidStats << s.getFailures() << ";0";  // Invalid + Pool switches
    poolAddresses << m_farm.get_pool_addresses();
    invalidStats << ";0;0";  // DualMining not supported

    int gpuIndex = 0;
    int numGpus = p.minersHashes.size();
    for (auto const& i : p.minersHashes)
    {
        detailedMhEth << std::fixed << std::setprecision(0) << (p.minerRate(i) / 1000.0f)
                      << (((numGpus - 1) > gpuIndex) ? ";" : "");
        detailedMhDcr << "off"
                      << (((numGpus - 1) > gpuIndex) ? ";" : "");  // DualMining not supported
        gpuIndex++;
    }

    gpuIndex = 0;
    numGpus = p.minerMonitors.size();
    for (auto const& i : p.minerMonitors)
    {
        tempAndFans << i.tempC << ";" << i.fanP
                    << (((numGpus - 1) > gpuIndex) ? ";" : "");  // Fetching Temp and Fans
        gpuIndex++;
    }

    Json::Value jRes;

    jRes[0] = ethminer_get_buildinfo()->project_name_with_version;  // miner version.
    jRes[1] = toString(runningTime.count());                        // running time, in minutes.
    jRes[2] = totalMhEth.str();  // total ETH hashrate in MH/s, number of ETH shares, number of ETH
                                 // rejected shares.
    jRes[3] = detailedMhEth.str();  // detailed ETH hashrate for all GPUs.
    jRes[4] = totalMhDcr.str();  // total DCR hashrate in MH/s, number of DCR shares, number of DCR
                                 // rejected shares.
    jRes[5] = detailedMhDcr.str();  // detailed DCR hashrate for all GPUs.
    jRes[6] = tempAndFans.str();    // Temperature and Fan speed(%) pairs for all GPUs.
    jRes[7] =
        poolAddresses.str();  // current mining pool. For dual mode, there will be two pools here.
    jRes[8] = invalidStats.str();  // number of ETH invalid shares, number of ETH pool switches,
                                   // number of DCR invalid shares, number of DCR pool switches.

    return jRes;
}

Json::Value ApiConnection::getMinerStatHR()
{
    // TODO:give key-value format
    auto runningTime = std::chrono::duration_cast<std::chrono::minutes>(
        steady_clock::now() - this->m_farm.farmLaunched());

    SolutionStats s = m_farm.getSolutionStats();
    WorkingProgress p = m_farm.miningProgress(true, true);

    ostringstream version;
    ostringstream runtime;
    Json::Value detailedMhEth;
    Json::Value detailedMhDcr;
    Json::Value temps;
    Json::Value fans;
    Json::Value powers;
    Json::Value ispaused;
    ostringstream poolAddresses;

    version << ethminer_get_buildinfo()->project_name_with_version;
    runtime << toString(runningTime.count());
    poolAddresses << m_farm.get_pool_addresses();

    assert(p.minersHashes.size() == p.minerMonitors.size());
    assert(p.minersHashes.size() == p.miningIsPaused.size());

    for (unsigned gpuIndex = 0; gpuIndex < p.minersHashes.size(); gpuIndex++)
    {
        auto const& minerhashes = p.minersHashes[gpuIndex];
        auto const& minermonitors = p.minerMonitors[gpuIndex];
        auto const& miningispaused = p.miningIsPaused[gpuIndex];

        detailedMhEth[gpuIndex] = (p.minerRate(minerhashes));
        // detailedMhDcr[gpuIndex] = "off"; //Not supported

        temps[gpuIndex] = minermonitors.tempC;    // Fetching Temps
        fans[gpuIndex] = minermonitors.fanP;      // Fetching Fans
        powers[gpuIndex] = minermonitors.powerW;  // Fetching Power

        ispaused[gpuIndex] = (bool)miningispaused;
    }

    Json::Value jRes;
    jRes["version"] = version.str();  // miner version.
    jRes["runtime"] = runtime.str();  // running time, in minutes.
    // total ETH hashrate in MH/s, number of ETH shares, number of ETH rejected shares.
    jRes["ethhashrate"] = (p.rate());
    jRes["ethhashrates"] = detailedMhEth;
    jRes["ethshares"] = s.getAccepts();
    jRes["ethrejected"] = s.getRejects();
    jRes["ethinvalid"] = s.getFailures();
    jRes["ethpoolsw"] = 0;
    // Hardware Info
    jRes["temperatures"] = temps;   // Temperatures(C) for all GPUs
    jRes["fanpercentages"] = fans;  // Fans speed(%) for all GPUs
    jRes["powerusages"] = powers;   // Power Usages(W) for all GPUs
    jRes["pooladdrs"] =
        poolAddresses.str();  // current mining pool. For dual mode, there will be two pools here.
    jRes["ispaused"] = ispaused;  // Is mining on GPU paused

    return jRes;
}
