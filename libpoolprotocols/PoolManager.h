#pragma once

#include <iostream>

#include <json/json.h>

#include <libdevcore/Worker.h>
#include <libethcore/Farm.h>
#include <libethcore/Miner.h>

#include "PoolClient.h"

using namespace std;

namespace dev
{
namespace eth
{

class PoolManager
{
public:
    PoolManager(boost::asio::io_service& io_service, PoolClient* client, Farm& farm,
        MinerType const& minerType, unsigned maxTries, unsigned failovertimeout);
    void addConnection(URI& conn);
    void clearConnections();
    Json::Value getConnectionsJson();
    void setActiveConnection(unsigned int idx);
    URI getActiveConnectionCopy();
    void removeConnection(unsigned int idx);
    void start();
    void stop();
    bool isConnected() { return p_client->isConnected(); };
    bool isRunning() { return m_running; };
    double getCurrentDifficulty();

private:
    void suspendMining();

    unsigned m_hashrateReportingTime = 60;
    unsigned m_hashrateReportingTimePassed = 0;
    unsigned m_failoverTimeout =
        0;  // After this amount of time in minutes of mining on a failover pool return to "primary"

    void check_failover_timeout(const boost::system::error_code& ec);

    std::atomic<bool> m_running = {false};
    void workLoop();

    unsigned m_connectionAttempt = 0;
    unsigned m_maxConnectionAttempts = 0;
    std::string m_lastConnectedHost = ""; // gets set when a connection has been established

    std::vector<URI> m_connections;
    unsigned m_activeConnectionIdx = 0;
    mutable Mutex m_activeConnectionMutex;

    std::thread m_workThread;

    h256 m_lastBoundary = h256();

    boost::asio::io_service::strand m_io_strand;
    boost::asio::deadline_timer m_failovertimer;
    PoolClient* p_client;
    Farm& m_farm;
    MinerType m_minerType;

    int m_lastEpoch = 0;
    double m_lastDifficulty = 0.0;
};
}  // namespace eth
}  // namespace dev
