#pragma once

#include <thread>

#include <libethcore/Farm.h>
#include <libpoolprotocols/PoolManager.h>

class httpServer
{
public:
    void run(string address, uint16_t port, bool show_hwmonitors, bool show_power);
    void run_thread();
    void getstat1(stringstream& ss);

    std::string m_port;

private:
    void tableHeader(stringstream& ss);
    bool m_show_hwmonitors;
    bool m_show_power;
};

extern httpServer http_server;
