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
    PoolManager(PoolClient* client, MinerType const& minerType, unsigned maxTries,
        unsigned failovertimeout);
    static PoolManager& p() { return *m_this; }
    void addConnection(URI& conn);
    void clearConnections();
    Json::Value getConnectionsJson();
    int setActiveConnection(unsigned int idx);
    URI getActiveConnectionCopy();
    int removeConnection(unsigned int idx);
    void start();
    void stop();
    bool isConnected() { return p_client->isConnected(); };
    bool isRunning() { return m_running; };
    double getCurrentDifficulty();
    unsigned getConnectionSwitches();
    unsigned getEpochChanges();

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
    std::atomic<unsigned> m_connectionSwitches = {0};

    std::vector<URI> m_connections;
    unsigned m_activeConnectionIdx = 0;
    mutable Mutex m_activeConnectionMutex;

    std::thread m_workThread;

    h256 m_lastBoundary = h256();

    boost::asio::io_service::strand m_io_strand;
    boost::asio::deadline_timer m_failovertimer;
    PoolClient* p_client;
    MinerType m_minerType;

    int m_lastEpoch = 0;
    std::atomic<unsigned> m_epochChanges = {0};
    double m_lastDifficulty = 0.0;

    static PoolManager* m_this;
};

}  // namespace eth
}  // namespace dev

