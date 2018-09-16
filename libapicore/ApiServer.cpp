#include "ApiServer.h"

#include <ethminer/buildinfo.h>

#include <libethcore/Farm.h>

#ifndef HOST_NAME_MAX
#define HOST_NAME_MAX 255
#endif

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

ApiServer::ApiServer(string address, int portnum, string password)
  : m_password(std::move(password)),
    m_address(address),
    m_acceptor(g_io_service),
    m_io_strand(g_io_service)
{
    if (portnum < 0)
    {
        m_portnumber = -portnum;
        m_readonly = true;
    }
    else
    {
        m_portnumber = portnum;
        m_readonly = false;
    }
}

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
        cwarn << "Could not start API server on port: " +
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
    auto session = std::make_shared<ApiConnection>(++lastSessionId, m_readonly, m_password);
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
        cnote << "New API session from " << session->socket().remote_endpoint();
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

    else if (_method == "miner_getstatdetail")
    {
        jResponse["result"] = getMinerStatDetail();
    }

    else if (_method == "miner_shuffle")
    {
        // Gives nonce scrambler a new range
        cnote << "Miner Shuffle requested";
        jResponse["result"] = true;
        Farm::f().shuffle();
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
        Farm::f().restart_async();
    }

    else if (_method == "miner_reboot")
    {
        if (!checkApiWriteAccess(m_readonly, jResponse))
            return;
        cnote << "Miner reboot requested";

        jResponse["result"] = Farm::f().reboot({{"api_miner_reboot"}});
    }

    else if (_method == "miner_getconnections")
    {
        // Returns a list of configured pools
        jResponse["result"] = PoolManager::p().getConnectionsJson();
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
            Json::Value pools = PoolManager::p().getConnectionsJson();
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
            PoolManager::p().addConnection(uri);
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

        if (PoolManager::p().setActiveConnection(index))
        {
            jResponse["error"]["code"] = -422;
            jResponse["error"]["message"] = "Index out of bounds";
            return;
        }

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

        int r;
        r = PoolManager::p().removeConnection(index);
        if (r == -1)
        {
            jResponse["error"]["code"] = -422;
            jResponse["error"]["message"] = "Index out of bounds";
            return;
        }
        if (r == -2)
        {
            jResponse["error"]["code"] = -460;
            jResponse["error"]["message"] = "Can't delete active connection";
            return;
        }

        jResponse["result"] = true;
    }

    else if (_method == "miner_getscramblerinfo")
    {
        jResponse["result"] = Farm::f().get_nonce_scrambler_json();
    }

    else if (_method == "miner_setscramblerinfo")
    {
        if (!checkApiWriteAccess(m_readonly, jResponse))
            return;

        Json::Value jRequestParams;
        if (!getRequestValue("params", jRequestParams, jRequest, false, jResponse))
            return;

        bool any_value_provided = false;
        uint64_t nonce = Farm::f().get_nonce_scrambler();
        unsigned exp = Farm::f().get_segment_width();

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
        Farm::f().set_nonce_scrambler(nonce);
        Farm::f().set_nonce_segment_width(exp);
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

        WorkingProgress p = Farm::f().miningProgress();
        if (index >= p.miningIsPaused.size())
        {
            jResponse["error"]["code"] = -422;
            jResponse["error"]["message"] = "Index out of bounds";
            return;
        }

        auto const& miner = Farm::f().getMiner(index);
        if (miner)
        {
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
        else
        {
            jResponse["result"] = false;
        }

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

        if (verbosity >= LOG_NEXT)
        {
            jResponse["error"]["code"] = -422;
            jResponse["error"]["message"] =
                "Verbosity out of bounds (0-" + to_string(LOG_NEXT - 1) + ")";
            return;
        }
        cnote << "Setting verbosity level to " << verbosity;
        g_logOptions = verbosity;
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
        steady_clock::now() - Farm::f().farmLaunched());
    auto connection = PoolManager::p().getActiveConnectionCopy();
    SolutionStats s = Farm::f().getSolutionStats();
    WorkingProgress p = Farm::f().miningProgress();

    ostringstream totalMhEth;
    ostringstream totalMhDcr;
    ostringstream detailedMhEth;
    ostringstream detailedMhDcr;
    ostringstream tempAndFans;
    ostringstream poolAddresses;
    ostringstream invalidStats;

    totalMhEth << std::fixed << std::setprecision(0) << p.hashRate / 1000.0f << ";"
               << s.getAccepts() << ";" << s.getRejects();
    totalMhDcr << "0;0;0";                    // DualMining not supported
    invalidStats << s.getFailures() << ";0";  // Invalid + Pool switches
    poolAddresses << connection.Host() << ':' << connection.Port();
    invalidStats << ";0;0";  // DualMining not supported

    int gpuIndex = 0;
    int numGpus = p.minersHashRates.size();
    for (auto const& i : p.minersHashRates)
    {
        detailedMhEth << std::fixed << std::setprecision(0) << i / 1000.0f
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
        steady_clock::now() - Farm::f().farmLaunched());
    auto connection = PoolManager::p().getActiveConnectionCopy();
    SolutionStats s = Farm::f().getSolutionStats();
    WorkingProgress p = Farm::f().miningProgress();

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
    poolAddresses << connection.Host() << ':' << connection.Port();

    assert(p.minersHashRates.size() == p.minerMonitors.size() || p.minerMonitors.size() == 0);
    assert(p.minersHashRates.size() == p.miningIsPaused.size());

    for (unsigned gpuIndex = 0; gpuIndex < p.minersHashRates.size(); gpuIndex++)
    {
        bool doMonitors = (gpuIndex < p.minerMonitors.size());

        detailedMhEth[gpuIndex] = p.minersHashRates[gpuIndex];
        // detailedMhDcr[gpuIndex] = "off"; //Not supported

        if (doMonitors)
        {
            temps[gpuIndex] = p.minerMonitors[gpuIndex].tempC;
            fans[gpuIndex] = p.minerMonitors[gpuIndex].fanP;
            powers[gpuIndex] = p.minerMonitors[gpuIndex].powerW;
        }
        else
        {
            temps[gpuIndex] = 0;
            fans[gpuIndex] = 0;
            powers[gpuIndex] = 0;
        }

        ispaused[gpuIndex] = (bool)p.miningIsPaused[gpuIndex];
    }

    Json::Value jRes;
    jRes["version"] = version.str();  // miner version.
    jRes["runtime"] = runtime.str();  // running time, in minutes.
    // total ETH hashrate in MH/s, number of ETH shares, number of ETH rejected shares.
    jRes["ethhashrate"] = uint64_t(p.hashRate);
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


Json::Value ApiConnection::getMinerStatDetailPerMiner(
    const WorkingProgress& p, const SolutionStats& s, size_t index)
{
    Json::Value jRes;
    auto const& miner = Farm::f().getMiner(index);

    jRes["index"] = (unsigned)index;

    /* Hash & Share infos */
    if (index < p.minersHashRates.size())
        jRes["hashrate"] = (uint64_t)p.minersHashRates[index];
    else
        jRes["hashrate"] = Json::Value::null;

    Json::Value jshares;
    jshares["accepted"] = s.getAccepts(index);
    jshares["rejected"] = s.getRejects(index);
    jshares["invalid"] = s.getFailures(index);
    jshares["acceptedstale"] = s.getAcceptedStales(index);
    auto solution_lastupdated = std::chrono::duration_cast<std::chrono::minutes>(
        std::chrono::steady_clock::now() - s.getLastUpdated(index));
    jshares["lastupdate"] = uint64_t(solution_lastupdated.count()); // last update of this gpu stat was x minutes ago
    jRes["shares"] = jshares;


    /* Hardware */
    if (index < p.minerMonitors.size())
    {
        jRes["temp"] = p.minerMonitors[index].tempC;
        jRes["fan"] = p.minerMonitors[index].fanP;
        jRes["power"] = p.minerMonitors[index].powerW;
    }
    else
    {
        jRes["temp"] = jRes["fan"] = jRes["power"] = Json::Value::null;
    }

    // TODO: PCI ID, Name, ... (some more infos - see listDevices())

    /* Pause infos */
    if (miner && index < p.miningIsPaused.size())
    {
        MinigPauseReason pause_reason = miner->get_mining_paused();
        MiningPause m;
        jRes["ispaused"] = m.is_mining_paused(pause_reason);
        jRes["pause_reason"] = m.get_mining_paused_string(pause_reason);
    }
    else
    {
        jRes["ispaused"] = jRes["pause_reason"] = Json::Value::null;
    }

    /* Nonce infos */
    auto segment_width = Farm::f().get_segment_width();
    uint64_t gpustartnonce = Farm::f().get_nonce_scrambler() + ((uint64_t)index << segment_width);
    jRes["nonce_start"] = gpustartnonce;
    jRes["nonce_stop"] = uint64_t(gpustartnonce + (1LL << segment_width));

    return jRes;
}


/**
 * @brief Return a total and per GPU detailed list of current status
 * As we return here difficulty and share counts (which are not getting resetted if we
 * switch pool) the results may "lie".
 * Eg: Calculating runtime, (current) difficulty and submitted shares must not match the hashrate.
 * Inspired by Andrea Lanfranchi comment on issue 1232:
 *    https://github.com/ethereum-mining/ethminer/pull/1232#discussion_r193995891
 * @return The json result
 */
Json::Value ApiConnection::getMinerStatDetail()
{
    auto runningTime = std::chrono::duration_cast<std::chrono::minutes>(
        std::chrono::steady_clock::now() - Farm::f().farmLaunched());

    SolutionStats s = Farm::f().getSolutionStats();
    WorkingProgress p = Farm::f().miningProgress();
    WorkPackage w = Farm::f().work();

    // ostringstream version;
    Json::Value gpus;
    Json::Value jRes;

    jRes["version"] = ethminer_get_buildinfo()->project_name_with_version;  // miner version.
    jRes["runtime"] = uint64_t(runningTime.count());  // running time, in minutes.

    {
        // Even the client should know which host was queried
        char hostName[HOST_NAME_MAX + 1];
        if (!gethostname(hostName, HOST_NAME_MAX + 1))
            jRes["hostname"] = hostName;
        else
            jRes["hostname"] = Json::Value::null;
    }

    /* connection info */
    auto connection = PoolManager::p().getActiveConnectionCopy();
    Json::Value jconnection;
    jconnection["uri"] = connection.String();
    // jconnection["endpoint"] = PoolManager::p().getClient()->ActiveEndPoint();
    jconnection["isconnected"] = PoolManager::p().isConnected();
    jconnection["switched"] = PoolManager::p().getConnectionSwitches();
    jRes["connection"] = jconnection;

    /* Pool info */
    jRes["difficulty"] = PoolManager::p().getCurrentDifficulty();
    if (w)
        jRes["epoch"] = w.epoch;
    else
        jRes["epoch"] = Json::Value::null;
    jRes["epoch_changes"] = PoolManager::p().getEpochChanges();

    /* basic setup */
    auto tstop = Farm::f().get_tstop();
    if (tstop)
    {
        jRes["tstart"] = Farm::f().get_tstart();
        jRes["tstop"] = tstop;
    }
    else
    {
        jRes["tstart"] = jRes["tstop"] = Json::Value::null;
    }

    /* gpu related info */
    if (Farm::f().getMiners().size())
    {
        for (size_t i = 0; i < Farm::f().getMiners().size(); i++)
        {
            jRes["gpus"].append(getMinerStatDetailPerMiner(p, s, i));
        }
    }
    else
    {
        jRes["gpus"] = Json::Value::null;
    }

    // total ETH hashrate
    jRes["hashrate"] = uint64_t(p.hashRate);

    // share information
    Json::Value jshares;
    jshares["accepted"] = s.getAccepts();
    jshares["rejected"] = s.getRejects();
    jshares["invalid"] = s.getFailures();
    jshares["acceptedstale"] = s.getAcceptedStales();
    jRes["shares"] = jshares;

    return jRes;
}
