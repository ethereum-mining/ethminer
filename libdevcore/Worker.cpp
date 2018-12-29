/*
    This file is part of ethminer.

    ethminer is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    ethminer is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with ethminer.  If not, see <http://www.gnu.org/licenses/>.
*/
/** @file Worker.cpp
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 */

#include <chrono>
#include <thread>

#include "Log.h"
#include "Worker.h"

using namespace std;
using namespace dev;

void Worker::startWorking()
{
    DEV_BUILD_LOG_PROGRAMFLOW(cnote, "Worker::startWorking() begin");

    // Can't start an already started thread
    if (m_state.load(memory_order_relaxed) != WorkerState::Stopped)
        return;

    m_state.store(WorkerState::Starting, memory_order_relaxed);

    m_work.reset(new thread([&]() {

        setThreadName(m_name.c_str());

        WorkerState ex = WorkerState::Starting;
        bool returnedError = false;

        if (m_state.compare_exchange_strong(ex, WorkerState::Started))
        {
            try
            {
                workLoop();
            }
            catch (std::exception const& _e)
            {
                returnedError = true;
                clog(WarnChannel) << "Exception thrown in Worker thread: " << _e.what();
            }
        }

        m_state.store(WorkerState::Stopped, memory_order_relaxed);

        if (returnedError && g_exitOnError)
        {
            clog(WarnChannel) << "Terminating due to --exit";
            raise(SIGTERM);
        }

    }));


    while (m_state == WorkerState::Starting)
        this_thread::sleep_for(chrono::microseconds(20));
    DEV_BUILD_LOG_PROGRAMFLOW(cnote, "Worker::startWorking() end");
}

void Worker::stopWorking()
{
    WorkerState ex = WorkerState::Started;
    m_state.compare_exchange_strong(ex, WorkerState::Stopping);
}

Worker::~Worker()
{
    DEV_BUILD_LOG_PROGRAMFLOW(cnote, "Worker::~Worker() begin");
    if (m_work->joinable())
    {
        m_state.store(WorkerState::Stopping, memory_order_relaxed);
        m_work->join();
        m_work.reset();
    }
    DEV_BUILD_LOG_PROGRAMFLOW(cnote, "Worker::~Worker() end");
}
