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
/** @file Log.h
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 *
 * The logging subsystem.
 */

#pragma once

#include <chrono>
#include <ctime>

#include "Common.h"
#include "CommonData.h"
#include "FixedHash.h"
#include "Terminal.h"
#include "vector_ref.h"

/// The logging system's current verbosity.
#define LOG_JSON 1
#define LOG_PER_GPU 2
#define LOG_CONNECT 32
#define LOG_SWITCH 64
#define LOG_SUBMIT 128
#define LOG_PROGRAMFLOW 256
#define LOG_NEXT 512

#if DEV_BUILD
#define DEV_BUILD_LOG_PROGRAMFLOW(_S, _V) \
    if (g_logOptions & LOG_PROGRAMFLOW)   \
    {                                     \
        _S << _V;                         \
    }                                     \
    ((void)(0))
#else
#define DEV_BUILD_LOG_PROGRAMFLOW(_S, _V) ((void)(0))
#endif

extern unsigned g_logOptions;
extern bool g_logNoColor;
extern bool g_logSyslog;
extern bool g_logStdout;

namespace dev
{
/// A simple log-output function that prints log messages to stdout.
void simpleDebugOut(std::string const&);

/// Set the current thread's log name.
void setThreadName(char const* _n);

/// Set the current thread's log name.
std::string getThreadName();

/// The default logging channels. Each has an associated verbosity and three-letter prefix (name()
/// ). Channels should inherit from LogChannel and define name() and verbosity.
struct LogChannel
{
    static const char* name();
};
struct WarnChannel : public LogChannel
{
    static const char* name();
};
struct NoteChannel : public LogChannel
{
    static const char* name();
};

class LogOutputStreamBase
{
public:
    LogOutputStreamBase(char const* _id);

    template <class T>
    void append(T const& _t)
    {
        m_sstr << _t;
    }

protected:
    std::stringstream m_sstr;  ///< The accrued log entry.
};

/// Logging class, iostream-like, that can be shifted to.
template <class Id>
class LogOutputStream : LogOutputStreamBase
{
public:
    /// Construct a new object.
    /// If _term is true the the prefix info is terminated with a ']' character; if not it ends only
    /// with a '|' character.
    LogOutputStream() : LogOutputStreamBase(Id::name()) {}

    /// Destructor. Posts the accrued log entry to the g_logPost function.
    ~LogOutputStream() { simpleDebugOut(m_sstr.str()); }

    /// Shift arbitrary data to the log. Spaces will be added between items as required.
    template <class T>
    LogOutputStream& operator<<(T const& _t)
    {
        append(_t);
        return *this;
    }
};

#define clog(X) dev::LogOutputStream<X>()

// Simple cout-like stream objects for accessing common log channels.
// Dirties the global namespace, but oh so convenient...
#define cnote clog(dev::NoteChannel)
#define cwarn clog(dev::WarnChannel)

}  // namespace dev
