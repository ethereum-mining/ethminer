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

#include "Log.h"

#include <map>
#include <thread>

#ifdef __APPLE__
#include <pthread.h>
#endif
#include "Guards.h"
using namespace std;
using namespace dev;

//⊳⊲◀▶■▣▢□▷◁▧▨▩▲◆◉◈◇◎●◍◌○◼☑☒☎☢☣☰☀♽♥♠✩✭❓✔✓✖✕✘✓✔✅⚒⚡⦸⬌∅⁕«««»»»⚙

// Logging
int g_logVerbosity = 5;
bool g_logNoColor = false;
bool g_logSyslog = false;

const char* LogChannel::name() { return EthGray ".."; }
const char* WarnChannel::name() { return EthRed " X"; }
const char* NoteChannel::name() { return EthBlue " i"; }

LogOutputStreamBase::LogOutputStreamBase(char const* _id, unsigned _v):
	m_verbosity(_v)
{
	if ((int)_v <= g_logVerbosity)
	{
		if (g_logSyslog)
			m_sstr << std::left << std::setw(8) << getThreadName() << " " EthReset;
		else {
			time_t rawTime = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
			char buf[24];
			if (strftime(buf, 24, "%X", localtime(&rawTime)) == 0)
				buf[0] = '\0'; // empty if case strftime fails
			m_sstr << _id << " " EthViolet << buf << " " EthNavy << std::left << std::setw(8) << getThreadName() << " " EthReset;
		}
	}
}

/// Associate a name with each thread for nice logging.
struct ThreadLocalLogName
{
	ThreadLocalLogName(char const* _name) { name = _name; }
	thread_local static char const* name;
};

thread_local char const* ThreadLocalLogName::name;

ThreadLocalLogName g_logThreadName("main");

string dev::getThreadName()
{
#if defined(__linux__) || defined(__APPLE__)
	char buffer[128];
	pthread_getname_np(pthread_self(), buffer, 127);
	buffer[127] = 0;
	return buffer;
#else
	return ThreadLocalLogName::name ? ThreadLocalLogName::name : "<unknown>";
#endif
}

void dev::setThreadName(char const* _n)
{
#if defined(__linux__)
	pthread_setname_np(pthread_self(), _n);
#elif defined(__APPLE__)
	pthread_setname_np(_n);
#else
	ThreadLocalLogName::name = _n;
#endif
}

void dev::simpleDebugOut(std::string const& _s)
{
	if (!g_logNoColor)
	{
		std::cerr << _s + '\n';
		return;
	}
	bool skip = false;
	std::stringstream ss;
	for (auto it : _s)
	{
		if (!skip && it == '\x1b')
			skip = true;
		else if (skip && it == 'm')
			skip = false;
		else if (!skip)
			ss << it;
	}
	ss << '\n';
	std::cerr << ss.str();
}
