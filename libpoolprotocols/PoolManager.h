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
        unsigned failovertimeout, unsigned ergodicity, bool reportHashrate);
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
    int getCurrentEpoch();
    double getCurrentDifficulty();
    unsigned getConnectionSwitches();
    unsigned getEpochChanges();

private:

    void rotateConnect();
    void showEpoch();
    void showDifficulty();

    unsigned m_hrReportingInterval = 60;
    unsigned m_failoverTimeout =
        0;  // After this amount of time in minutes of mining on a failover pool return to "primary"

    void failovertimer_elapsed(const boost::system::error_code& ec);
    void submithrtimer_elapsed(const boost::system::error_code& ec);

    std::atomic<bool> m_running = {false};
    std::atomic<bool> m_stopping = {false};

    bool m_hashrate;           // Whether or not submit hashrate to work provider (pool)
    std::string m_hashrateId;  // The unique client Id to use when submitting hashrate
    unsigned m_ergodicity = 0;
    unsigned m_connectionAttempt = 0;
    unsigned m_maxConnectionAttempts = 0;
    std::string m_selectedHost = "";  // Holds host name (and endpoint) of selected connection
    std::atomic<unsigned> m_connectionSwitches = {0};

    std::vector<URI> m_connections;
    unsigned m_activeConnectionIdx = 0;
    mutable Mutex m_activeConnectionMutex;

    WorkPackage m_currentWp;

    boost::asio::io_service::strand m_io_strand;
    boost::asio::deadline_timer m_failovertimer;
    boost::asio::deadline_timer m_submithrtimer;

    PoolClient* p_client;
    MinerType m_minerType;

    std::atomic<unsigned> m_epochChanges = {0};

    static PoolManager* m_this;
};

}  // namespace eth
}  // namespace dev

