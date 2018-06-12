#include "ApiServer.h"

#include <ethminer-buildinfo.h>

ApiServer::ApiServer(
    boost::asio::io_service& io_service, int portnum, bool readonly, string password, Farm& f, PoolManager& mgr)
  : m_readonly(readonly),
    m_password(std::move(password)),
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

    tcp::endpoint endpoint(tcp::v4(), m_portnumber);

    // Try to bind to port number
    // if exception occurs it may be due to the fact that
    // requested port is already in use by another service
    try
    {
        m_acceptor.open(endpoint.protocol());
        m_acceptor.bind(endpoint);
        m_acceptor.listen(64);
    }
    catch (const std::exception& _e)
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

void ApiConnection::processRequest(Json::Value& requestObject)
{
	Json::Value jRes;
	jRes["jsonrpc"] = "2.0";

	// Strict sanity checks over jsonrpc v2
	if (
		(!requestObject.isMember("jsonrpc") || requestObject["jsonrpc"].empty() || !requestObject["jsonrpc"].isString() || requestObject.get("jsonrpc", ".") != "2.0") ||
		(!requestObject.isMember("method") || requestObject["method"].empty() || !requestObject["method"].isString()) ||
		(!requestObject.isMember("id") || requestObject["id"].empty() || !requestObject["id"].isUInt())
		)
	{
		jRes["id"] = Json::nullValue;
		jRes["error"]["code"] = -32600;
		jRes["error"]["message"] = "Invalid Request";
		sendSocketData(jRes);
		return;
	}


	// Process messages
	std::string _method = requestObject.get("method", "").asString();
	jRes["id"] = requestObject.get("id", 0).asInt();

    // Check authentication
    if (!m_is_authenticated)
    {
        if (_method == "api_authorize")
        {
            if (!requestObject.isMember("params") || requestObject["params"].empty() ||
                !requestObject["params"].isObject())
            {
                jRes["error"]["code"] = -32600;
                jRes["error"]["message"] = "Invalid request";
            }
            else
            {
                Json::Value jPrm = requestObject["params"];
                if (!jPrm.isMember("psw") || jPrm["psw"].empty() ||
                    !jPrm["psw"].isString())
                {
                    jRes["error"]["code"] = -32602;
                    jRes["error"]["message"] = "Missing password";
                }
                else
                {
                    if (jPrm.get("psw", "").asString() == m_password)
                    {
                        m_is_authenticated = true;
                    }
                    else
                    {
						// Use error code like http 401 Unauthorized
                        jRes["error"]["code"] = -401;
                        jRes["error"]["message"] = "Invalid password";
                        cerr << "Invalid API password provided.";
                    }
                }
            }
        }
        else
        {
			// Use error code like http 403 Forbidden
            jRes["error"]["code"] = -403;
            jRes["error"]["message"] = "Authorization needed";
        }
    }

    if (m_is_authenticated)
    {
        if (_method == "miner_getstat1")
        {
            jRes["result"] = getMinerStat1();
        }
        else if (_method == "miner_getstathr")
        {
            jRes["result"] = getMinerStatHR();
        }
        else if (_method == "miner_shuffle")
        {
            // Gives nonce scrambler a new range
            cnote << "Miner Shuffle requested";
            jRes["result"] = true;
            m_farm.shuffle();
        }
        else if (_method == "miner_ping")
        {
            // Replies back to (check for liveness)
            jRes["result"] = "pong";
        }
        else if (_method == "miner_restart")
        {
            // Send response to client of success
            // and invoke an async restart
            // to prevent locking
            if (m_readonly)
            {
                jRes["error"]["code"] = -32601;
                jRes["error"]["message"] = "Method not available";
            }
            else
            {
                cnote << "Miner Restart requested";
                jRes["result"] = true;
                m_farm.restart_async();
            }
        }
        else if (_method == "miner_reboot")
        {
            // Not implemented yet
            jRes["error"]["code"] = -32601;
            jRes["error"]["message"] = "Method not implemented";
        }
		else if (_method == "miner_getconnections")
		{
			// Returns a list of configured pools
			jRes["result"] = m_mgr.getConnectionsJson();
		}
		else if (_method == "miner_addpool")
		{
			if (m_readonly)
			{
				jRes["error"]["code"] = -32601;
				jRes["error"]["message"] = "Method not available";
			}
			else
			{

				if (!requestObject.isMember("params") || requestObject["params"].empty() ||
					!requestObject["params"].isObject())
				{
					jRes["error"]["code"] = -32600;
					jRes["error"]["message"] = "Invalid request";
				}
				else
				{
					Json::Value jPrm = requestObject["params"];
					if (!jPrm.isMember("uri") || jPrm["uri"].empty() ||
						!jPrm["uri"].isString())
					{
						jRes["error"]["code"] = -32602;
						jRes["error"]["message"] = "Missing pool uri";
					}
					else
					{
						dev::URI uri;
						try
						{
							uri = jPrm.get("uri", "").asString();
							if (!uri.KnownScheme())
							{
								jRes["error"]["code"] = -422;
								jRes["error"]["message"] = ("Unknown URI scheme " + uri.Scheme());
							} 
							else
							{
								// Check other pools already present share the same scheme family (stratum or getwork)
								Json::Value pools = m_mgr.getConnectionsJson();
								for(Json::Value::ArrayIndex i = 0; i != pools.size(); i++)
								{
									dev::URI poolUri = pools[i]["uri"].asString();
									if (uri.Family() != poolUri.Family()) {
										jRes["error"]["code"] = -422;
										jRes["error"]["message"] = "Mixed stratum and getwork endpoints not supported.";
										break;
									}
								}

								// If everything ok then add this new uri
								if (!jRes.isMember("error"))
								{
									m_mgr.addConnection(uri);
									jRes["result"] = true;
								}

							}
						}
						catch (...)
						{
							jRes["error"]["code"] = -422;
							jRes["error"]["message"] = "Bad URI";
						}

					}
				}
			}
		}
		else if (_method == "miner_setactiveconnection")
		{
			if (m_readonly)
			{
				jRes["error"]["code"] = -32601;
				jRes["error"]["message"] = "Method not available";
			}
			else
			{

				if (!requestObject.isMember("params") || requestObject["params"].empty() ||
					!requestObject["params"].isObject())
				{
					jRes["error"]["code"] = -32600;
					jRes["error"]["message"] = "Invalid request";
				}
				else
				{
					Json::Value jPrm = requestObject["params"];
					if (!jPrm.isMember("index") || jPrm["index"].empty() ||
						!jPrm["index"].isInt())
					{
						jRes["error"]["code"] = -32602;
						jRes["error"]["message"] = "Missing pool index";
					}
					else
					{
						Json::Value pools = m_mgr.getConnectionsJson();
						if (jPrm["index"].asInt() < 0 || Json::ArrayIndex(jPrm["index"].asInt()) >= pools.size()) 
						{
							jRes["error"]["code"] = -422;
							jRes["error"]["message"] = "Index out of bounds";
						}
						else
						{
							m_mgr.removeConnection(jPrm["index"].asInt());
							jRes["result"] = true;
						}
					}
				}
			}

		}
		else if (_method == "miner_removeconnection")
		{
			if (m_readonly)
			{
				jRes["error"]["code"] = -32601;
				jRes["error"]["message"] = "Method not available";
			}
			else
			{

				if (!requestObject.isMember("params") || requestObject["params"].empty() ||
					!requestObject["params"].isObject())
				{
					jRes["error"]["code"] = -32600;
					jRes["error"]["message"] = "Invalid request";
				}
				else
				{
					Json::Value jPrm = requestObject["params"];
					if (!jPrm.isMember("index") || jPrm["index"].empty() ||
						!jPrm["index"].isInt())
					{
						jRes["error"]["code"] = -32602;
						jRes["error"]["message"] = "Missing pool index";
					}
					else
					{
						Json::Value pools = m_mgr.getConnectionsJson();
						if (jPrm["index"].asInt() < 0 || Json::ArrayIndex(jPrm["index"].asInt()) >= pools.size())
						{
							jRes["error"]["code"] = -422;
							jRes["error"]["message"] = "Index out of bounds";
						}
						else
						{
							Json::ArrayIndex i(jPrm["index"].asInt());
							if (pools[i]["active"].asBool()) {

								jRes["error"]["code"] = -460;
								jRes["error"]["message"] = "Can't delete active connection";

							}
							else
							{
								m_mgr.setActiveConnection(jPrm["index"].asInt());
								jRes["result"] = true;
							}
						}
					}
				}
			}

		}
		else if (_method == "miner_getscramblerinfo")
		{

			jRes["result"] = m_farm.get_nonce_scrambler_json();

		}
		else if (_method == "miner_setscramblerinfo")
		{
			if (m_readonly)
			{
				jRes["error"]["code"] = -32601;
				jRes["error"]["message"] = "Method not available";
			}
			else
			{

				if (!requestObject.isMember("params") || requestObject["params"].empty() ||
					!requestObject["params"].isObject())
				{
					jRes["error"]["code"] = -32600;
					jRes["error"]["message"] = "Invalid request";
				}
				else
				{
					Json::Value jPrm = requestObject["params"];
					if (!jPrm.isMember("noncescrambler") && !jPrm.isMember("segmentwidth")) 
					{
						jRes["error"]["code"] = -32602;
						jRes["error"]["message"] = "Missing parameters";
					}
					else
					{
						uint64_t nonce = m_farm.get_nonce_scrambler();
						unsigned exp = m_farm.get_segment_width();

						try
						{
							if (!jPrm["noncescrambler"].empty())
							{
								nonce = jPrm["noncescrambler"].asLargestUInt();
							}
							if (!jPrm["segmentwidth"].empty())
							{
								exp = jPrm["segmentwidth"].asUInt();
							}
						}
						catch (...)
						{
							jRes["error"]["code"] = -422;
							jRes["error"]["message"] = "Bad values";
						}

						if (!jRes.isMember("error"))
						{
							if (exp < 10) exp = 10;		// Not below 
							if (exp > 50) exp = 40;		// Not above 
							m_farm.set_nonce_scrambler(nonce);
							m_farm.set_nonce_segment_width(exp);
							jRes["result"] = true;
						}

					}
				}
			}

		}
        else
        {
            // Any other method not found
            jRes["error"]["code"] = -32601;
            jRes["error"]["message"] = "Method not found";
        }
    }

    // Send response
    sendSocketData(jRes);
}

void ApiConnection::recvSocketData()
{
    // cnote << "ApiConnection::recvSocketData";
    boost::asio::async_read_until(m_socket, m_recvBuffer, "\n",
        m_io_strand.wrap(boost::bind(&ApiConnection::onRecvSocketDataCompleted, this,
            boost::asio::placeholders::error, boost::asio::placeholders::bytes_transferred)));
}

void ApiConnection::onRecvSocketDataCompleted(const boost::system::error_code& ec, std::size_t bytes_transferred)
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
                Json::Reader jRdr;
                if (jRdr.parse(message, jMsg))
                {
                    processRequest(jMsg);
                }
                else
                {
                    Json::Value jRes;
                    jRes["jsonrpc"] = "2.0";
                    jRes["id"] = Json::nullValue;
                    jRes["error"]["code"] = -32700;
                    jRes["error"]["message"] = "Parse Error";
                    sendSocketData(jRes);
                }
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

void ApiConnection::sendSocketData(Json::Value const & jReq) {

    if (!m_socket.is_open())
        return;

    std::ostream os(&m_sendBuffer);
    os << m_jWriter.write(jReq);  // Do not add lf. It's added by writer.

    async_write(m_socket, m_sendBuffer,
        m_io_strand.wrap(boost::bind(
            &ApiConnection::onSendSocketDataCompleted, this, boost::asio::placeholders::error)));
}

void ApiConnection::onSendSocketDataCompleted(const boost::system::error_code& ec) {

	if (ec) disconnect();

}

Json::Value ApiConnection::getMinerStat1()
{
	
	auto runningTime = std::chrono::duration_cast<std::chrono::minutes>(steady_clock::now() - this->m_farm.farmLaunched());
	
	SolutionStats s = m_farm.getSolutionStats();
	WorkingProgress p = m_farm.miningProgress(true);
	
	ostringstream totalMhEth; 
	ostringstream totalMhDcr; 
	ostringstream detailedMhEth;
	ostringstream detailedMhDcr;
	ostringstream tempAndFans;
	ostringstream poolAddresses;
	ostringstream invalidStats;
	
	totalMhEth << std::fixed << std::setprecision(0) << (p.rate() / 1000.0f) << ";" << s.getAccepts() << ";" << s.getRejects();
	totalMhDcr << "0;0;0"; // DualMining not supported
	invalidStats << s.getFailures() << ";0"; // Invalid + Pool switches
    poolAddresses << m_farm.get_pool_addresses(); 
	invalidStats << ";0;0"; // DualMining not supported
	
	int gpuIndex = 0;
	int numGpus = p.minersHashes.size();
	for (auto const& i: p.minersHashes)
	{
		detailedMhEth << std::fixed << std::setprecision(0) << (p.minerRate(i) / 1000.0f) << (((numGpus -1) > gpuIndex) ? ";" : "");
		detailedMhDcr << "off" << (((numGpus -1) > gpuIndex) ? ";" : ""); // DualMining not supported
		gpuIndex++;
	}

	gpuIndex = 0;
	numGpus = p.minerMonitors.size();
	for (auto const& i : p.minerMonitors)
	{
		tempAndFans << i.tempC << ";" << i.fanP << (((numGpus - 1) > gpuIndex) ? ";" : ""); // Fetching Temp and Fans
		gpuIndex++;
	}

	Json::Value jRes;

	jRes[0] = ethminer_get_buildinfo()->project_version;  //miner version.
	jRes[1] = toString(runningTime.count()); // running time, in minutes.
	jRes[2] = totalMhEth.str();              // total ETH hashrate in MH/s, number of ETH shares, number of ETH rejected shares.
	jRes[3] = detailedMhEth.str();           // detailed ETH hashrate for all GPUs.
	jRes[4] = totalMhDcr.str();              // total DCR hashrate in MH/s, number of DCR shares, number of DCR rejected shares.
	jRes[5] = detailedMhDcr.str();           // detailed DCR hashrate for all GPUs.
	jRes[6] = tempAndFans.str();             // Temperature and Fan speed(%) pairs for all GPUs.
	jRes[7] = poolAddresses.str();           // current mining pool. For dual mode, there will be two pools here.
	jRes[8] = invalidStats.str();            // number of ETH invalid shares, number of ETH pool switches, number of DCR invalid shares, number of DCR pool switches.

	return jRes;
}

Json::Value ApiConnection::getMinerStatHR()
{
	
	//TODO:give key-value format
	auto runningTime = std::chrono::duration_cast<std::chrono::minutes>(steady_clock::now() - this->m_farm.farmLaunched());
	
	SolutionStats s = m_farm.getSolutionStats();
	WorkingProgress p = m_farm.miningProgress(true,true);
	
	ostringstream version; 
	ostringstream runtime; 
	Json::Value detailedMhEth;
	Json::Value detailedMhDcr;
	Json::Value temps;
	Json::Value fans;
	Json::Value powers;
	ostringstream poolAddresses;
	
	version << ethminer_get_buildinfo()->project_version;
	runtime << toString(runningTime.count());
    poolAddresses << m_farm.get_pool_addresses(); 
	
	int gpuIndex = 0;
	for (auto const& i: p.minersHashes)
	{
		detailedMhEth[gpuIndex] = (p.minerRate(i));
		//detailedMhDcr[gpuIndex] = "off"; //Not supported
		gpuIndex++;
	}

	gpuIndex = 0;
	for (auto const& i : p.minerMonitors)
	{
		temps[gpuIndex] = i.tempC ; // Fetching Temps 
		fans[gpuIndex] = i.fanP; // Fetching Fans
		powers[gpuIndex] =  i.powerW; // Fetching Power
		gpuIndex++;
	}

	Json::Value jRes;

	jRes["version"] = version.str();		// miner version.
	jRes["runtime"] = runtime.str();		// running time, in minutes.
	// total ETH hashrate in MH/s, number of ETH shares, number of ETH rejected shares.
	jRes["ethhashrate"] = (p.rate());
	jRes["ethhashrates"] = detailedMhEth;
	jRes["ethshares"] 	= s.getAccepts();
	jRes["ethrejected"] = s.getRejects();
	jRes["ethinvalid"] 	= s.getFailures();
	jRes["ethpoolsw"] 	= 0;
	// Hardware Info
	jRes["temperatures"] = temps;             		// Temperatures(C) for all GPUs
	jRes["fanpercentages"] = fans;             		// Fans speed(%) for all GPUs
	jRes["powerusages"] = powers;         			// Power Usages(W) for all GPUs
	jRes["pooladdrs"] = poolAddresses.str();        // current mining pool. For dual mode, there will be two pools here.

	return jRes;

}

