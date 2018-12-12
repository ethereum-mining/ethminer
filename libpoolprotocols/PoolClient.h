#pragma once

#include <queue>

#include <boost/asio/ip/address.hpp>

#include <libethcore/Miner.h>
#include <libpoolprotocols/PoolURI.h>

extern boost::asio::io_service g_io_service;

using namespace std;

namespace dev
{
namespace eth
{
struct Session
{
    // Tstamp of sessio start
    chrono::steady_clock::time_point start = chrono::steady_clock::now();
    // Whether or not worker is subscribed
    atomic<bool> subscribed = {false};
    // Whether or not worker is authorized
    atomic<bool> authorized = {false};
    // Total duration of session in minutes
    unsigned long duration()
    {
        return (chrono::duration_cast<chrono::minutes>(chrono::steady_clock::now() - start))
            .count();
    }

    // EthereumStratum (1 and 2)

    // Extranonce currently active
    uint64_t extraNonce = 0;
    // Length of extranonce in bytes
    unsigned int extraNonceSizeBytes = 0;
    // Next work target
    h256 nextWorkBoundary =
        h256("0x00000000ffff0000000000000000000000000000000000000000000000000000");

    // EthereumStratum (2 only)
    bool firstMiningSet = false;
    unsigned int timeout = 30;  // Default to 30 seconds
    string sessionId = "";
    string workerId = "";
    string algo = "ethash";
    unsigned int epoch = 0;
    chrono::steady_clock::time_point lastTxStamp = chrono::steady_clock::now();

};

class PoolClient
{
public:
    virtual ~PoolClient() noexcept = default;

    // Sets the connection definition to be used by the client
    void setConnection(std::shared_ptr<URI> _conn)
    {
        m_conn = _conn;
        m_conn->Responds(false);
    }

    // Gets a pointer to the currently active connection definition
    std::shared_ptr<URI> getConnection() { return m_conn; }

    // Releases the pointer to the connection definition
    void unsetConnection() { m_conn = nullptr; }

    virtual void connect() = 0;
    virtual void disconnect() = 0;
    virtual void submitHashrate(uint64_t const& rate, string const& id) = 0;
    virtual void submitSolution(const Solution& solution) = 0;
    virtual bool isConnected() { return m_connected.load(memory_order_relaxed); }
    virtual bool isPendingState() { return false; }

    virtual bool isSubscribed()
    {
        return (m_session ? m_session->subscribed.load(memory_order_relaxed) : false);
    }
    virtual bool isAuthorized()
    {
        return (m_session ? m_session->authorized.load(memory_order_relaxed) : false);
    }

    virtual string ActiveEndPoint()
    {
        return (m_connected.load(memory_order_relaxed) ? " [" + toString(m_endpoint) + "]" : "");
    }

    using SolutionAccepted = function<void(chrono::milliseconds const&, unsigned const&, bool)>;
    using SolutionRejected = function<void(chrono::milliseconds const&, unsigned const&)>;
    using Disconnected = function<void()>;
    using Connected = function<void()>;
    using WorkReceived = function<void(WorkPackage const&)>;

    void onSolutionAccepted(SolutionAccepted const& _handler) { m_onSolutionAccepted = _handler; }
    void onSolutionRejected(SolutionRejected const& _handler) { m_onSolutionRejected = _handler; }
    void onDisconnected(Disconnected const& _handler) { m_onDisconnected = _handler; }
    void onConnected(Connected const& _handler) { m_onConnected = _handler; }
    void onWorkReceived(WorkReceived const& _handler) { m_onWorkReceived = _handler; }

protected:
    unique_ptr<Session> m_session = nullptr;

    std::atomic<bool> m_connected = {false};  // This is related to socket ! Not session

    boost::asio::ip::basic_endpoint<boost::asio::ip::tcp> m_endpoint;

    std::shared_ptr<URI> m_conn = nullptr;

    SolutionAccepted m_onSolutionAccepted;
    SolutionRejected m_onSolutionRejected;
    Disconnected m_onDisconnected;
    Connected m_onConnected;
    WorkReceived m_onWorkReceived;
};
}  // namespace eth
}  // namespace dev
