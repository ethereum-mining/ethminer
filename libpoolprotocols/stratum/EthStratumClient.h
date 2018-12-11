#pragma once

#include <iostream>

#include <boost/array.hpp>
#include <boost/asio.hpp>
#include <boost/asio/ssl.hpp>
#include <boost/bind.hpp>
#include <boost/lockfree/queue.hpp>

#include <json/json.h>

#include <libdevcore/FixedHash.h>
#include <libdevcore/Log.h>
#include <libethcore/EthashAux.h>
#include <libethcore/Farm.h>
#include <libethcore/Miner.h>

#include "../PoolClient.h"

using namespace std;
using namespace dev;
using namespace dev::eth;

template <typename Verifier>
class verbose_verification
{
public:
    verbose_verification(Verifier verifier) : verifier_(verifier) {}

    bool operator()(bool preverified, boost::asio::ssl::verify_context& ctx)
    {
        char subject_name[256];
        X509* cert = X509_STORE_CTX_get_current_cert(ctx.native_handle());
        X509_NAME_oneline(X509_get_subject_name(cert), subject_name, 256);
        bool verified = verifier_(preverified, ctx);
#ifdef DEV_BUILD
        cnote << "Certificate: " << subject_name << " " << (verified ? "Ok" : "Failed");
#else
        if (!verified)
            cnote << "Certificate: " << subject_name << " "
                  << "Failed";
#endif
        return verified;
    }

private:
    Verifier verifier_;
};

class EthStratumClient : public PoolClient
{
public:
    enum StratumProtocol
    {
        STRATUM = 0,
        ETHPROXY,
        ETHEREUMSTRATUM,
        ETHEREUMSTRATUM2
    };

    EthStratumClient(int worktimeout, int responsetimeout);

    void init_socket();
    void connect() override;
    void disconnect() override;

    // Connected and Connection Statuses
    bool isConnected() override
    {
        bool _ret = PoolClient::isConnected();
        return _ret && !isPendingState();
    }
    bool isPendingState() override
    {
        return (m_connecting.load(std::memory_order_relaxed) ||
                m_disconnecting.load(std::memory_order_relaxed));
    }

    void submitHashrate(uint64_t const& rate, string const& id) override;
    void submitSolution(const Solution& solution) override;

    h256 currentHeaderHash() { return m_current.header; }
    bool current() { return static_cast<bool>(m_current); }

private:
    void startSession();
    void disconnect_finalize();
    void enqueue_response_plea();
    std::chrono::milliseconds dequeue_response_plea();
    void clear_response_pleas();
    void resolve_handler(
        const boost::system::error_code& ec, boost::asio::ip::tcp::resolver::iterator i);
    void start_connect();
    void connect_handler(const boost::system::error_code& ec);
    void workloop_timer_elapsed(const boost::system::error_code& ec);

    void processResponse(Json::Value& responseObject);
    std::string processError(Json::Value& erroresponseObject);
    void processExtranonce(std::string& enonce);

    void recvSocketData();
    void onRecvSocketDataCompleted(
        const boost::system::error_code& ec, std::size_t bytes_transferred);
    void send(Json::Value const& jReq);
    void sendSocketData();
    void onSendSocketDataCompleted(const boost::system::error_code& ec);
    void onSSLShutdownCompleted(const boost::system::error_code& ec);

    std::atomic<bool> m_disconnecting = {false};
    std::atomic<bool> m_connecting = {false};
    std::atomic<bool> m_authpending = {false};

    // seconds to trigger a work_timeout (overwritten in constructor)
    int m_worktimeout;

    // seconds timeout for responses and connection (overwritten in constructor)
    int m_responsetimeout;

    // default interval for workloop timer (milliseconds)
    int m_workloop_interval = 1000;

    WorkPackage m_current;
    std::chrono::time_point<std::chrono::steady_clock> m_current_timestamp;

    boost::asio::io_service& m_io_service;  // The IO service reference passed in the constructor
    boost::asio::io_service::strand m_io_strand;
    boost::asio::ip::tcp::socket* m_socket;
    std::string m_message;  // The internal message string buffer
    bool m_newjobprocessed = false;

    // Use shared ptrs to avoid crashes due to async_writes
    // see
    // https://stackoverflow.com/questions/41526553/can-async-write-cause-segmentation-fault-when-this-is-deleted
    std::shared_ptr<boost::asio::ssl::stream<boost::asio::ip::tcp::socket>> m_securesocket;
    std::shared_ptr<boost::asio::ip::tcp::socket> m_nonsecuresocket;

    boost::asio::streambuf m_sendBuffer;
    boost::asio::streambuf m_recvBuffer;
    Json::StreamWriterBuilder m_jSwBuilder;

    boost::asio::deadline_timer m_workloop_timer;

    std::atomic<int> m_response_pleas_count = {0};
    std::atomic<std::chrono::steady_clock::duration> m_response_plea_older;
    boost::lockfree::queue<std::chrono::steady_clock::time_point> m_response_plea_times;

    std::atomic<bool> m_txPending = {false};
    boost::lockfree::queue<std::string*> m_txQueue;

    boost::asio::ip::tcp::resolver m_resolver;
    std::queue<boost::asio::ip::basic_endpoint<boost::asio::ip::tcp>> m_endpoints;

    unsigned m_solution_submitted_max_id;  // maximum json id we used to send a solution

    ///@brief Auxiliary function to make verbose_verification objects.
    template <typename Verifier>
    verbose_verification<Verifier> make_verbose_verification(Verifier verifier)
    {
        return verbose_verification<Verifier>(verifier);
    }
};
