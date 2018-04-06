#include <map>
#include <boost/optional/optional_io.hpp>
#include <boost/algorithm/string.hpp>
#include <network/uri/detail/decode.hpp>
#include <libpoolprotocols/PoolURI.h>

#include <iostream>

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
	{"http",		  {ProtocolFamily::GETWORK, SecureLevel::NONE,  0}}
};

URI::URI(const std::string uri)
{
	std::string u = uri;
	if (u.find("://") == std::string::npos)
		u = std::string("unspecified://") + u;
	m_uri = network::uri(u);
}

bool URI::KnownScheme()
{
	if (!m_uri.scheme())
		return false;
	std::string s(*m_uri.scheme());
	boost::trim(s);
	return s_schemes.find(s) != s_schemes.end();
}

ProtocolFamily URI::ProtoFamily() const
{
	if (!m_uri.scheme())
		return ProtocolFamily::STRATUM;
	std::string s(*m_uri.scheme());
	s = network::detail::decode(s);
	boost::trim(s);
	return s_schemes[s].family;
}

unsigned URI::ProtoVersion() const
{
	if (!m_uri.scheme())
		return 0;
	std::string s(*m_uri.scheme());
	s = network::detail::decode(s);
	boost::trim(s);
	return s_schemes[s].version;
}

SecureLevel URI::ProtoSecureLevel() const
{
	if (!m_uri.scheme())
		return SecureLevel::NONE;
	std::string s(*m_uri.scheme());
	s = network::detail::decode(s);
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
	if (!m_uri.scheme())
		return "";
	std::string s(*m_uri.scheme());
	s = network::detail::decode(s);
	boost::trim(s);
	return s;
}

std::string URI::Host() const
{
	if (!m_uri.host())
		return "";
	std::string s(*m_uri.host());
	s = network::detail::decode(s);
	boost::trim(s);
	return s;
}

std::string URI::Path() const
{
	if (!m_uri.path())
		return "";
	std::string s(*m_uri.path());
	s = network::detail::decode(s);
	boost::trim(s);
	return s;
}

unsigned short URI::Port() const
{
	if (!m_uri.port())
		return 0;
	std::string s(*m_uri.port());
	s = network::detail::decode(s);
	boost::trim(s);
	return (unsigned short)atoi(s.c_str());
}

std::string URI::User() const
{
	if (!m_uri.user_info())
		return "";
	std::string s(*m_uri.user_info());
	s = network::detail::decode(s);
	boost::trim(s);
	size_t f = s.find(":");
	if (f == std::string::npos)
		return s;
	return s.substr(0, f);
}

std::string URI::Pswd() const
{
	if (!m_uri.user_info())
		return "";
	std::string s(*m_uri.user_info());
	s = network::detail::decode(s);
	boost::trim(s);
	size_t f = s.find(":");
	if (f == std::string::npos)
		return "";
	return s.substr(f + 1);
}

