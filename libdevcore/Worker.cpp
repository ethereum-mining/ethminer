/*
	This file is part of cpp-ethereum.

	cpp-ethereum is free software: you can redistribute it and/or modify
	it under the terms of the GNU General Public License as published by
	the Free Software Foundation, either version 3 of the License, or
	(at your option) any later version.

	cpp-ethereum is distributed in the hope that it will be useful,
	but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	GNU General Public License for more details.

	You should have received a copy of the GNU General Public License
	along with cpp-ethereum.  If not, see <http://www.gnu.org/licenses/>.
*/
/** @file Worker.cpp
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 */

#include "Worker.h"

#include <chrono>
#include <thread>
#include "Log.h"
using namespace std;
using namespace dev;

void Worker::startWorking()
{
	cdebug << "startWorking for thread" << m_name;
	Guard l(x_work);
	if (m_work)
		return;
	cdebug << "Spawning" << m_name;
	m_stop = false;
	m_work.reset(new thread([&]()
	{
		setThreadName(m_name.c_str());
		while (!m_stop)
		{
			this_thread::sleep_for(chrono::milliseconds(100));
			doWork();
		}
		cdebug << "Finishing up worker thread";
		doneWorking();
	}));
}

void Worker::stopWorking()
{
	cdebug << "stopWorking for thread" << m_name;
	Guard l(x_work);
	if (!m_work)
		return;
	cdebug << "Stopping" << m_name;
	m_stop = true;
	m_work->join();
	m_work.reset();
	cdebug << "Stopped" << m_name;
}

