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
/** @file Log.h
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 *
 * The logging subsystem.
 */

#pragma once

#include <ctime>
#include <chrono>
#include <boost/thread.hpp>
#include "vector_ref.h"

namespace eth
{

/// The null output stream. Used when logging is disabled.
class NullOutputStream
{
public:
	template <class T> NullOutputStream& operator<<(T const&) { return *this; }
};

/// A simple log-output function that prints log messages to stdout.
void simpleDebugOut(std::string const&, char const* );

/// The logging system's current verbosity.
extern int g_logVerbosity;

/// The current method that the logging system uses to output the log messages. Defaults to simpleDebugOut().
extern std::function<void(std::string const&, char const*)> g_logPost;

/// Map of Log Channel types to bool, false forces the channel to be disabled, true forces it to be enabled.
/// If a channel has no entry, then it will output as long as its verbosity (LogChannel::verbosity) is less than
/// or equal to the currently output verbosity (g_logVerbosity).
extern std::map<std::type_info const*, bool> g_logOverride;

/// Associate a name with each thread for nice logging.
struct ThreadLocalLogName
{
	ThreadLocalLogName(std::string _name) { m_name.reset(new std::string(_name)); };
	boost::thread_specific_ptr<std::string> m_name;
};

/// The current thread's name.
extern ThreadLocalLogName t_logThreadName;

/// Set the current thread's log name.
inline void setThreadName(char const* _n) { t_logThreadName.m_name.reset(new std::string(_n)); }

/// The default logging channels. Each has an associated verbosity and three-letter prefix (name() ).
/// Channels should inherit from LogChannel and define name() and verbosity.
struct LogChannel { static const char* name() { return "   "; } static const int verbosity = 1; };
struct LeftChannel: public LogChannel  { static const char* name() { return "<<<"; } };
struct RightChannel: public LogChannel { static const char* name() { return ">>>"; } };
struct WarnChannel: public LogChannel  { static const char* name() { return "!!!"; } static const int verbosity = 0; };
struct NoteChannel: public LogChannel  { static const char* name() { return "***"; } };
struct DebugChannel: public LogChannel { static const char* name() { return "---"; } static const int verbosity = 0; };

/// Logging class, iostream-like, that can be shifted to.
template <class Id, bool _AutoSpacing = true>
class LogOutputStream
{
public:
	/// Construct a new object.
	/// If _term is true the the prefix info is terminated with a ']' character; if not it ends only with a '|' character.
	LogOutputStream(bool _term = true)
	{
		std::type_info const* i = &typeid(Id);
		auto it = g_logOverride.find(i);
		if ((it != g_logOverride.end() && it->second == true) || (it == g_logOverride.end() && Id::verbosity <= g_logVerbosity))
		{
			time_t rawTime = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
			char buf[24];
			if (strftime(buf, 24, "%X", localtime(&rawTime)) == 0)
				buf[0] = '\0'; // empty if case strftime fails
			m_sstr << Id::name() << " [ " << buf << " | " << *(t_logThreadName.m_name.get()) << (_term ? " ] " : "");
		}
	}

	/// Destructor. Posts the accrued log entry to the g_logPost function.
	~LogOutputStream() { if (Id::verbosity <= g_logVerbosity) g_logPost(m_sstr.str(), Id::name()); }

	/// Shift arbitrary data to the log. Spaces will be added between items as required.
	template <class T> LogOutputStream& operator<<(T const& _t) { if (Id::verbosity <= g_logVerbosity) { if (_AutoSpacing && m_sstr.str().size() && m_sstr.str().back() != ' ') m_sstr << " "; m_sstr << _t; } return *this; }

private:
	std::stringstream m_sstr;	///< The accrued log entry.
};

// Simple cout-like stream objects for accessing common log channels.
// Dirties the global namespace, but oh so convenient...
#define cnote eth::LogOutputStream<eth::NoteChannel, true>()
#define cwarn eth::LogOutputStream<eth::WarnChannel, true>()

// Null stream-like objects.
#define ndebug if (true) {} else eth::NullOutputStream()
#define nlog(X) if (true) {} else eth::NullOutputStream()
#define nslog(X) if (true) {} else eth::NullOutputStream()

// Kill debugging log channel when we're in release mode.
#if NDEBUG
#define cdebug ndebug
#else
#define cdebug eth::LogOutputStream<eth::DebugChannel, true>()
#endif

// Kill all logs when when NLOG is defined.
#if NLOG
#define clog(X) nlog(X)
#define cslog(X) nslog(X)
#else
#define clog(X) eth::LogOutputStream<X, true>()
#define cslog(X) eth::LogOutputStream<X, false>()
#endif

}
