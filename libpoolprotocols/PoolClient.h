#pragma once

#include <queue>

#include <boost/asio/ip/address.hpp>

#include <libethcore/Farm.h>
#include <libethcore/Miner.h>
#include <libpoolprotocols/PoolURI.h>

using namespace std;

namespace dev
{
namespace eth
{
class PoolClient
{
public:
    virtual ~PoolClient() noexcept = default;


    void setConnection(URI* conn)
    {
        m_conn = conn;
        m_canconnect.store(false, std::memory_order_relaxed);
    }

    const URI* getConnection() { return m_conn; }
    void unsetConnection() { m_conn = nullptr; }

    virtual void connect() = 0;
    virtual void disconnect() = 0;

    virtual void submitHashrate(string const& rate) = 0;
    virtual void submitSolution(const Solution& solution) = 0;
    virtual bool isConnected() = 0;
    virtual bool isPendingState() = 0;
    virtual string ActiveEndPoint() = 0;

    using SolutionAccepted = std::function<void(bool const&, std::chrono::milliseconds const&, unsigned const& miner_index)>;
    using SolutionRejected = std::function<void(bool const&, std::chrono::milliseconds const&, unsigned const& miner_index)>;
    using Disconnected = std::function<void()>;
    using Connected = std::function<void()>;
    using WorkReceived = std::function<void(WorkPackage const&)>;

    void onSolutionAccepted(SolutionAccepted const& _handler) { m_onSolutionAccepted = _handler; }
    void onSolutionRejected(SolutionRejected const& _handler) { m_onSolutionRejected = _handler; }
    void onDisconnected(Disconnected const& _handler) { m_onDisconnected = _handler; }
    void onConnected(Connected const& _handler) { m_onConnected = _handler; }
    void onWorkReceived(WorkReceived const& _handler) { m_onWorkReceived = _handler; }

protected:
    std::atomic<bool> m_subscribed = {false};
    std::atomic<bool> m_authorized = {false};
    std::atomic<bool> m_connected = {false};
    std::atomic<bool> m_canconnect = {false};

    boost::asio::ip::basic_endpoint<boost::asio::ip::tcp> m_endpoint;

    URI* m_conn = nullptr;

    SolutionAccepted m_onSolutionAccepted;
    SolutionRejected m_onSolutionRejected;
    Disconnected m_onDisconnected;
    Connected m_onConnected;
    WorkReceived m_onWorkReceived;
};
}  // namespace eth
}  // namespace dev
