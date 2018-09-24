#pragma once

#include <thread>

#include <libethcore/Farm.h>
#include <libpoolprotocols/PoolManager.h>

class httpServer
{
public:
    void run(string address, uint16_t port, unsigned hwmonlvl);
    void run_thread();
    void getstat1(stringstream& ss);

    std::string m_port;

private:
    void tableHeader(stringstream& ss);
    unsigned m_hwmonlvl;
};

extern httpServer http_server;
