#include "EthGetworkClient.h"

#include <chrono>

#include <ethash/ethash.hpp>

using namespace std;
using namespace dev;
using namespace eth;

using boost::asio::ip::tcp;

EthGetworkClient::EthGetworkClient(int worktimeout, unsigned farmRecheckPeriod)
  : PoolClient(),
    m_farmRecheckPeriod(farmRecheckPeriod),
    m_io_strand(g_io_service),
    m_socket(g_io_service),
    m_resolver(g_io_service),
    m_endpoints(),
    m_getwork_timer(g_io_service),
    m_worktimeout(worktimeout)
{
    m_jSwBuilder.settings_["indentation"] = "";

    Json::Value jGetWork;
    jGetWork["id"] = unsigned(1);
    jGetWork["jsonrpc"] = "2.0";
    jGetWork["method"] = "eth_getWork";
    jGetWork["params"] = Json::Value(Json::arrayValue);
    m_jsonGetWork = std::string(Json::writeString(m_jSwBuilder, jGetWork));
}

EthGetworkClient::~EthGetworkClient()
{
    // Do not stop io service.
    // It's global
}

void EthGetworkClient::connect()
{
    // Prevent unnecessary and potentially dangerous recursion
    bool expected = false;
    if (!m_connecting.compare_exchange_strong(expected, true, memory_order::memory_order_relaxed))
        return;

    // Reset status flags
    m_getwork_timer.cancel();
    
    // Initialize a new queue of end points
    m_endpoints = std::queue<boost::asio::ip::basic_endpoint<boost::asio::ip::tcp>>();
    m_endpoint = boost::asio::ip::basic_endpoint<boost::asio::ip::tcp>();

    if (m_conn->HostNameType() == dev::UriHostNameType::Dns ||
        m_conn->HostNameType() == dev::UriHostNameType::Basic)
    {
        // Begin resolve all ips associated to hostname
        // calling the resolver each time is useful as most
        // load balancers will give Ips in different order
        m_resolver = boost::asio::ip::tcp::resolver(g_io_service);
        boost::asio::ip::tcp::resolver::query q(m_conn->Host(), toString(m_conn->Port()));

        // Start resolving async
        m_resolver.async_resolve(
            q, m_io_strand.wrap(boost::bind(&EthGetworkClient::handle_resolve, this,
                   boost::asio::placeholders::error, boost::asio::placeholders::iterator)));
    }
    else
    {
        // No need to use the resolver if host is already an IP address
        m_endpoints.push(boost::asio::ip::tcp::endpoint(
            boost::asio::ip::address::from_string(m_conn->Host()), m_conn->Port()));
        send(m_jsonGetWork);
    }
}

void EthGetworkClient::disconnect()
{
    // Release session
    m_connected.store(false, memory_order_relaxed);
    if (m_session)
    {
        m_conn->addDuration(m_session->duration());
    }
    m_session = nullptr;

    m_connecting.store(false, std::memory_order_relaxed);
    m_txPending.store(false, std::memory_order_relaxed);
    m_getwork_timer.cancel();

    m_txQueue.consume_all([](std::string* l) { delete l; });
    m_request.consume(m_request.capacity());
    m_response.consume(m_response.capacity());

    if (m_onDisconnected)
        m_onDisconnected();
}

void EthGetworkClient::begin_connect()
{
    if (!m_endpoints.empty())
    {
        // Pick the first endpoint in list.
        // Eventually endpoints get discarded on connection errors
        m_endpoint = m_endpoints.front();
        m_socket.async_connect(
            m_endpoint, m_io_strand.wrap(boost::bind(&EthGetworkClient::handle_connect, this, boost::placeholders::_1)));
    }
    else
    {
        cwarn << "No more IP addresses to try for host: " << m_conn->Host();
        disconnect();
    }
}

void EthGetworkClient::handle_connect(const boost::system::error_code& ec)
{
    if (!ec && m_socket.is_open())
    {

        // If in "connecting" phase raise the proper event
        if (m_connecting.load(std::memory_order_relaxed))
        {
            // Initialize new session
            m_connected.store(true, memory_order_relaxed);
            m_session = unique_ptr<Session>(new Session);
            m_session->subscribed.store(true, memory_order_relaxed);
            m_session->authorized.store(true, memory_order_relaxed);
            
            m_connecting.store(false, std::memory_order_relaxed);

            if (m_onConnected)
                m_onConnected();
            m_current_tstamp = std::chrono::steady_clock::now();
        }

        // Retrieve 1st line waiting in the queue and submit
        // if other lines waiting they will be processed 
        // at the end of the processed request
        Json::Reader jRdr;
        std::string* line;
        std::ostream os(&m_request);
        if (!m_txQueue.empty())
        {
            while (m_txQueue.pop(line))
            {
                if (line->size())
                {

                    jRdr.parse(*line, m_pendingJReq);
                    m_pending_tstamp = std::chrono::steady_clock::now();

                    // Make sure path begins with "/"
                    string _path = (m_conn->Path().empty() ? "/" : m_conn->Path());

                    os << "POST " << _path << " HTTP/1.0\r\n";
                    os << "Host: " << m_conn->Host() << "\r\n";
                    os << "Content-Type: application/json"
                       << "\r\n";
                    os << "Content-Length: " << line->length() << "\r\n";
                    os << "Connection: close\r\n\r\n";  // Double line feed to mark the
                                                        // beginning of body
                    // The payload
                    os << *line;

                    // Out received message only for debug purpouses
                    if (g_logOptions & LOG_JSON)
                        cnote << " >> " << *line;

                    delete line;

                    async_write(m_socket, m_request,
                        m_io_strand.wrap(boost::bind(&EthGetworkClient::handle_write, this,
                            boost::asio::placeholders::error)));
                    break;
                }
                delete line;
            }
        }
        else
        {
            m_txPending.store(false, std::memory_order_relaxed);
        }

    }
    else
    {
        if (ec != boost::asio::error::operation_aborted)
        {
            // This endpoint does not respond
            // Pop it and retry
            cwarn << "Error connecting to " << m_conn->Host() << ":" << toString(m_conn->Port())
                  << " : " << ec.message();
            m_endpoints.pop();
            begin_connect();
        }
    }
}

void EthGetworkClient::handle_write(const boost::system::error_code& ec)
{
    if (!ec)
    {
        // Transmission succesfully sent.
        // Read the response async. 
        async_read(m_socket, m_response, boost::asio::transfer_all(),
            m_io_strand.wrap(boost::bind(&EthGetworkClient::handle_read, this,
                boost::asio::placeholders::error, boost::asio::placeholders::bytes_transferred)));
    }
    else
    {
        if (ec != boost::asio::error::operation_aborted)
        {
            cwarn << "Error writing to " << m_conn->Host() << ":" << toString(m_conn->Port())
                  << " : " << ec.message();
            m_endpoints.pop();
            begin_connect();
        }
    }
}

void EthGetworkClient::handle_read(
    const boost::system::error_code& ec, std::size_t bytes_transferred)
{
    if (!ec || ec == boost::asio::error::eof)
    {
        // Close socket
        if (m_socket.is_open())
            m_socket.close();

        // Get the whole message
        std::string rx_message(
            boost::asio::buffer_cast<const char*>(m_response.data()), bytes_transferred);
        m_response.consume(bytes_transferred);

        // Empty response ?
        if (!rx_message.size())
        {
            cwarn << "Invalid response from " << m_conn->Host() << ":" << toString(m_conn->Port());
            disconnect();
            return;
        }

        // Read message by lines.
        // First line is http status
        // Other lines are headers
        // A double "\r\n" identifies begin of body
        // The rest is body
        std::string line;
        std::string linedelimiter = "\r\n";
        std::size_t delimiteroffset = rx_message.find(linedelimiter);

        unsigned int linenum = 0;
        bool isHeader = true;
        while (rx_message.length() && delimiteroffset != std::string::npos)
        {
            linenum++;
            line = rx_message.substr(0, delimiteroffset);
            rx_message.erase(0, delimiteroffset + 2);
            
            // This identifies the beginning of body
            if (line.empty())
            {
                isHeader = false;
                delimiteroffset = rx_message.find(linedelimiter);
                if (delimiteroffset != std::string::npos)
                    continue;
                boost::replace_all(rx_message, "\n", "");
                line = rx_message;
            }

            // Http status
            if (isHeader && linenum == 1)
            {
                if (line.substr(0, 7) != "HTTP/1.")
                {
                    cwarn << "Invalid response from " << m_conn->Host() << ":"
                          << toString(m_conn->Port());
                    disconnect();
                    return;
                }
                std::size_t spaceoffset = line.find(' ');
                if (spaceoffset == std::string::npos)
                {
                    cwarn << "Invalid response from " << m_conn->Host() << ":"
                          << toString(m_conn->Port());
                    disconnect();
                    return;
                }
                std::string status = line.substr(spaceoffset + 1);
                if (status.substr(0, 3) != "200")
                {
                    cwarn << m_conn->Host() << ":" << toString(m_conn->Port())
                          << " reported status " << status;
                    disconnect();
                    return;
                }
            }

            // Body
            if (!isHeader)
            {
                // Out received message only for debug purpouses
                if (g_logOptions & LOG_JSON)
                    cnote << " << " << line;

                // Test validity of chunk and process
                Json::Value jRes;
                Json::Reader jRdr;
                if (jRdr.parse(line, jRes))
                {
                    // Run in sync so no 2 different async reads may overlap
                    processResponse(jRes);
                }
                else
                {
                    string what = jRdr.getFormattedErrorMessages();
                    boost::replace_all(what, "\n", " ");
                    cwarn << "Got invalid Json message : " << what;
                }

            }

            delimiteroffset = rx_message.find(linedelimiter);
        }

        // Is there anything else in the queue
        if (!m_txQueue.empty())
        {
            begin_connect();
        }
        else
        {
            // Signal end of async send/receive operations
            m_txPending.store(false, std::memory_order_relaxed);
        }

    }
    else
    {
        if (ec != boost::asio::error::operation_aborted)
        {
            cwarn << "Error reading from :" << m_conn->Host() << ":" << toString(m_conn->Port())
                  << " : "
                  << ec.message();
            disconnect();
        }
       
    }
}

void EthGetworkClient::handle_resolve(
    const boost::system::error_code& ec, tcp::resolver::iterator i)
{
    if (!ec)
    {
        while (i != tcp::resolver::iterator())
        {
            m_endpoints.push(i->endpoint());
            i++;
        }
        m_resolver.cancel();

        // Resolver has finished so invoke connection asynchronously
        send(m_jsonGetWork);
    }
    else
    {
        cwarn << "Could not resolve host " << m_conn->Host() << ", " << ec.message();
        disconnect();
    }
}

void EthGetworkClient::processResponse(Json::Value& JRes) 
{
    unsigned _id = 0;  // This SHOULD be the same id as the request it is responding to 
    bool _isSuccess = false;  // Whether or not this is a succesful or failed response
    string _errReason = "";   // Content of the error reason

    if (!JRes.isMember("id"))
    {
        cwarn << "Missing id member in response from " << m_conn->Host() << ":"
              << toString(m_conn->Port());
        return;
    }
    // We get the id from pending jrequest
    // It's not guaranteed we get response labelled with same id
    // For instance Dwarfpool always responds with "id":0
    _id = m_pendingJReq.get("id", unsigned(0)).asUInt();
    _isSuccess = JRes.get("error", Json::Value::null).empty();
    _errReason = (_isSuccess ? "" : processError(JRes));

    // We have only theese possible ids
    // 0 or 1 as job notification
    // 9 as response for eth_submitHashrate
    // 40+ for responses to mining submissions
    if (_id == 0 || _id == 1)
    {
        // Getwork might respond with an error to
        // a request. (eg. node is still syncing)
        // In such case delay further requests
        // by 30 seconds.
        // Otherwise resubmit another getwork request
        // with a delay of m_farmRecheckPeriod ms.
        if (!_isSuccess)
        {
            cwarn << "Got " << _errReason << " from " << m_conn->Host() << ":"
                  << toString(m_conn->Port());
            m_getwork_timer.expires_from_now(boost::posix_time::seconds(30));
            m_getwork_timer.async_wait(
                m_io_strand.wrap(boost::bind(&EthGetworkClient::getwork_timer_elapsed, this,
                    boost::asio::placeholders::error)));
        }
        else
        {
            if (!JRes.isMember("result"))
            {
                cwarn << "Missing data for eth_getWork request from " << m_conn->Host() << ":"
                      << toString(m_conn->Port());
            }
            else
            {
                Json::Value JPrm = JRes.get("result", Json::Value::null);
                WorkPackage newWp;
                
                newWp.header = h256(JPrm.get(Json::Value::ArrayIndex(0), "").asString());
                newWp.seed = h256(JPrm.get(Json::Value::ArrayIndex(1), "").asString());
                newWp.boundary = h256(JPrm.get(Json::Value::ArrayIndex(2), "").asString());
                newWp.job = newWp.header.hex();
                if (m_current.header != newWp.header)
                {
                    m_current = newWp;
                    m_current_tstamp = std::chrono::steady_clock::now();

                    if (m_onWorkReceived)
                        m_onWorkReceived(m_current);
                }
                m_getwork_timer.expires_from_now(boost::posix_time::milliseconds(m_farmRecheckPeriod));
                m_getwork_timer.async_wait(
                    m_io_strand.wrap(boost::bind(&EthGetworkClient::getwork_timer_elapsed, this,
                        boost::asio::placeholders::error)));
            }
        }

    }
    else if (_id == 9)
    {
        // Response to hashrate submission
        // Actually don't do anything
    }
    else if (_id >= 40 && _id <= m_solution_submitted_max_id)
    {
        if (_isSuccess && JRes["result"].isConvertibleTo(Json::ValueType::booleanValue))
            _isSuccess = JRes["result"].asBool();

        std::chrono::milliseconds _delay = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - m_pending_tstamp);

        const unsigned miner_index = _id - 40;
        if (_isSuccess)
        {
            if (m_onSolutionAccepted)
                m_onSolutionAccepted(_delay, miner_index, false);
        }
        else
        {
            if (m_onSolutionRejected)
                m_onSolutionRejected(_delay, miner_index);
        }
    }

}

std::string EthGetworkClient::processError(Json::Value& JRes)
{
    std::string retVar;

    if (JRes.isMember("error") &&
        !JRes.get("error", Json::Value::null).isNull())
    {
        if (JRes["error"].isConvertibleTo(Json::ValueType::stringValue))
        {
            retVar = JRes.get("error", "Unknown error").asString();
        }
        else if (JRes["error"].isConvertibleTo(Json::ValueType::arrayValue))
        {
            for (auto i : JRes["error"])
            {
                retVar += i.asString() + " ";
            }
        }
        else if (JRes["error"].isConvertibleTo(Json::ValueType::objectValue))
        {
            for (Json::Value::iterator i = JRes["error"].begin(); i != JRes["error"].end(); ++i)
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

void EthGetworkClient::send(Json::Value const& jReq)
{
    send(std::string(Json::writeString(m_jSwBuilder, jReq)));
}

void EthGetworkClient::send(std::string const& sReq) 
{
    std::string* line = new std::string(sReq);
    m_txQueue.push(line);

    bool ex = false;
    if (m_txPending.compare_exchange_strong(ex, true, std::memory_order_relaxed))
        begin_connect();
}

void EthGetworkClient::submitHashrate(uint64_t const& rate, string const& id)
{
    // No need to check for authorization
    if (m_session)
    {
        Json::Value jReq;
        jReq["id"] = unsigned(9);
        jReq["jsonrpc"] = "2.0";
        jReq["method"] = "eth_submitHashrate";
        jReq["params"] = Json::Value(Json::arrayValue);
        jReq["params"].append(toHex(rate, HexPrefix::Add));  // Already expressed as hex
        jReq["params"].append(id);                           // Already prefixed by 0x
        send(jReq);
    }

}

void EthGetworkClient::submitSolution(const Solution& solution)
{

    if (m_session)
    {
        Json::Value jReq;
        string nonceHex = toHex(solution.nonce);

        unsigned id = 40 + solution.midx;
        jReq["id"] = id;
        jReq["jsonrpc"] = "2.0";
        m_solution_submitted_max_id = max(m_solution_submitted_max_id, id);
        jReq["method"] = "eth_submitWork";
        jReq["params"] = Json::Value(Json::arrayValue);
        jReq["params"].append("0x" + nonceHex);
        jReq["params"].append("0x" + solution.work.header.hex());
        jReq["params"].append("0x" + solution.mixHash.hex());
        send(jReq);
    }

}

void EthGetworkClient::getwork_timer_elapsed(const boost::system::error_code& ec) 
{
    // Triggers the resubmission of a getWork request
    if (!ec)
    {
        // Check if last work is older than timeout
        std::chrono::seconds _delay = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::steady_clock::now() - m_current_tstamp);
        if (_delay.count() > m_worktimeout)
        {
            cwarn << "No new work received in " << m_worktimeout << " seconds.";
            m_endpoints.pop();
            disconnect();
        }
        else
        {
            send(m_jsonGetWork);
        }

    }
}
