#pragma once

#include <string>
#include <network/uri.hpp>

// A simple URI parser specifically for mining pool endpoints
namespace dev
{

enum class SecureLevel {NONE = 0, TLS12, TLS};
enum class ProtocolFamily {GETWORK = 0, STRATUM};

class URI
{
public:
	URI() {};
	URI(const std::string uri);

	std::string	Scheme() const;
	std::string	Host() const;
	std::string	Path() const;
	unsigned short	Port() const;
	std::string	User() const;
	std::string	Pass() const;
	SecureLevel	SecLevel() const;
	ProtocolFamily	Family() const;
	unsigned	Version() const;

	bool		KnownScheme();

	static std::string KnownSchemes(ProtocolFamily family);

private:
	network::uri m_uri;
};

}
