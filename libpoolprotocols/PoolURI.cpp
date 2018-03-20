#include <map>
#include <boost/optional/optional_io.hpp>
#include <boost/algorithm/string.hpp>
#include <libpoolprotocols/PoolURI.h>

using namespace dev;

typedef struct {
	ProtocolFamily family;
	SecureLevel secure;
	unsigned version;
} SchemeAttributes;

static std::map<std::string, SchemeAttributes> s_schemes = {
	{"stratum+tcp",	  {ProtocolFamily::STRATUM, SecureLevel::NONE,  0}},
	{"stratum1+tcp",  {ProtocolFamily::STRATUM, SecureLevel::NONE,  1}},
	{"stratum2+tcp",  {ProtocolFamily::STRATUM, SecureLevel::NONE,  2}},
	{"stratum+tls",	  {ProtocolFamily::STRATUM, SecureLevel::TLS,   0}},
	{"stratum1+tls",  {ProtocolFamily::STRATUM, SecureLevel::TLS,   1}},
	{"stratum2+tls",  {ProtocolFamily::STRATUM, SecureLevel::TLS,   2}},
	{"stratum+tls12", {ProtocolFamily::STRATUM, SecureLevel::TLS12, 0}},
	{"stratum1+tls12",{ProtocolFamily::STRATUM, SecureLevel::TLS12, 1}},
	{"stratum2+tls12",{ProtocolFamily::STRATUM, SecureLevel::TLS12, 2}},
	{"stratum+ssl",	  {ProtocolFamily::STRATUM, SecureLevel::TLS12, 0}},
	{"stratum1+ssl",  {ProtocolFamily::STRATUM, SecureLevel::TLS12, 1}},
	{"stratum2+ssl",  {ProtocolFamily::STRATUM, SecureLevel::TLS12, 2}},
	{"http",	  {ProtocolFamily::GETWORK, SecureLevel::NONE,  0}}
};

URI::URI() {}

URI::URI(const std::string uri)
{
	std::string u = uri;
	if (u.find("://") == std::string::npos)
		u = std::string("unspecified://") + u;
	m_uri = network::uri(u);
}

bool URI::KnownScheme()
{
	std::string s(*m_uri.scheme());
	boost::trim(s);
	return s_schemes.find(s) != s_schemes.end();
}

ProtocolFamily URI::ProtoFamily() const
{
	std::string s(*m_uri.scheme());
	boost::trim(s);
	return s_schemes[s].family;
}

unsigned URI::ProtoVersion() const
{
	std::string s(*m_uri.scheme());
	boost::trim(s);
	return s_schemes[s].version;
}

SecureLevel URI::ProtoSecureLevel() const
{
	std::string s(*m_uri.scheme());
	boost::trim(s);
	return s_schemes[s].secure;
}

std::string URI::KnownSchemes(ProtocolFamily family)
{
	std::string schemes;
	for(const auto&s : s_schemes)
		if (s.second.family == family)
			schemes += s.first + " ";
	boost::trim(schemes);
	return schemes;
}

std::string URI::Scheme() const
{
	std::string s(*m_uri.scheme());
	boost::trim(s);
	return s;
}

std::string URI::Host() const
{
	std::string s(*m_uri.host());
	boost::trim(s);
	if (s == "--")
		return "";
	return s;
}

unsigned short URI::Port() const
{
	std::string s(*m_uri.port());
	boost::trim(s);
	if (s == "--")
		return 0;
	return (unsigned short)atoi(s.c_str());
}

std::string URI::User() const
{
	std::stringstream ss;
	std::string s(*m_uri.user_info());
	boost::trim(s);
	if (s == "--")
		return "";
	size_t f = s.find(":");
	if (f == std::string::npos)
		return s;
	return s.substr(0, f);
}

std::string URI::Pswd() const
{
	std::string s(*m_uri.user_info());
	boost::trim(s);
	if (s == "--")
		return "";
	size_t f = s.find(":");
	if (f == std::string::npos)
		return "";
	return s.substr(f + 1);
}
