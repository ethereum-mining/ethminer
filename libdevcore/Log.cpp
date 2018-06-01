#include <iostream>
#include <boost/core/null_deleter.hpp>
#include <boost/log/attributes/clock.hpp>
#include <boost/log/attributes/function.hpp>
#include <boost/log/expressions.hpp>
#include <boost/log/sinks/async_frontend.hpp>
#include <boost/log/sinks/text_ostream_backend.hpp>
#include <boost/log/sources/global_logger_storage.hpp>
#include <boost/log/sources/severity_channel_logger.hpp>
#include <boost/log/support/date_time.hpp>
#include <boost/log/utility/exception_handler.hpp>
#include "Log.h"

int g_logVerbosity = 5;

namespace logging = boost::log;
namespace src = logging::sources;
namespace expr = logging::expressions;
namespace keywords = logging::keywords;

namespace {
	// Associate a name with each thread for nice logging.
	struct ThreadLocalLogName
	{
		ThreadLocalLogName(std::string const& _name) { m_name.reset(new std::string(_name)); }
    	boost::thread_specific_ptr<std::string> m_name;
	};

	ThreadLocalLogName g_logThreadName("main");
}

BOOST_LOG_ATTRIBUTE_KEYWORD(severity, "Severity", logging::trivial::severity_level)
BOOST_LOG_ATTRIBUTE_KEYWORD(threadName, "ThreadName", std::string)
BOOST_LOG_ATTRIBUTE_KEYWORD(timestamp, "TimeStamp", boost::posix_time::ptime)

static void formatter(logging::record_view const& rec, logging::basic_formatting_ostream<char>& strm)
{

    strm << EthViolet;
    logging::basic_formatter<char> f =
		 expr::stream << expr::format_date_time<boost::posix_time::ptime>("TimeStamp", "%H:%M:%S ");
	f(rec, strm);
	strm
		<< EthReset;
	std::stringstream ss;
	ss << rec[logging::trivial::severity];
	std::string s = ss.str();
	const char *cp;
	if (s == "info")
		cp = EthGreen "info";
	else if (s == "warning")
		cp = EthYellow "warn";
	else if (s == "error")
		cp = EthRed "err ";
	else
		cp = EthYellow "fail";

	strm
    	<< cp << EthReset << ' '
    	<< EthNavy << std::setw(8) << std::left << logging::extract<std::string>("ThreadName", rec) << EthReset " "
    	<< rec[expr::smessage];
}

void setupLogging()
{
    auto sink = boost::make_shared<logging::sinks::asynchronous_sink<logging::sinks::text_ostream_backend>>();

    boost::shared_ptr<std::ostream> stream{&std::cout, boost::null_deleter{}};
    sink->locked_backend()->add_stream(stream);

    sink->set_formatter(&formatter);

    logging::core::get()->add_sink(sink);

    logging::core::get()->add_global_attribute(
        "ThreadName", logging::attributes::make_function(&getThreadName));
    logging::core::get()->add_global_attribute(
        "TimeStamp", logging::attributes::local_clock());

    logging::core::get()->set_exception_handler(
        logging::make_exception_handler<std::exception>([](std::exception const& _ex) {
        std::cerr << "Exception from the logging library: " << _ex.what() << '\n';
    }));
}

std::string getThreadName() {
#if defined(__GLIBC__) || defined(__APPLE__)
    char buffer[128];
    pthread_getname_np(pthread_self(), buffer, 127);
    buffer[127] = 0;
    return buffer;
#else
    return g_logThreadName.m_name.get() ? *g_logThreadName.m_name.get() : "<unknown>";
#endif
}

void setThreadName(std::string const& _n) {
#if defined(__GLIBC__)
    pthread_setname_np(pthread_self(), _n.c_str());
#elif defined(__APPLE__)
    pthread_setname_np(_n.c_str());
#else
    g_logThreadName.m_name.reset(new std::string(_n));
#endif
}

