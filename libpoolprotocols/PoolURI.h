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
    URI(){};
    URI(const std::string uri);

    std::string Scheme() const;
    std::string Host() const;
    std::string Path() const;
    unsigned short Port() const;
    std::string User() const;
    std::string Pass() const;
    SecureLevel SecLevel() const;
    ProtocolFamily Family() const;
    unsigned Version() const;
    std::string string() { return m_uri.string(); }

    bool KnownScheme();

    static std::string KnownSchemes(ProtocolFamily family);

    void SetStratumMode(unsigned mode, bool confirmed)
    {
        m_stratumMode = mode;
        m_stratumModeConfirmed = confirmed;
    }
    void SetStratumMode(unsigned mode) { m_stratumMode = mode; }
    unsigned StratumMode() { return m_stratumMode; }
    bool StratumModeConfirmed() { return m_stratumModeConfirmed; }
    bool IsUnrecoverable() { return m_unrecoverable; }
    void MarkUnrecoverable() { m_unrecoverable = true; }

private:
    network::uri m_uri;
    bool m_stratumModeConfirmed = false;
    unsigned m_stratumMode = 999;  // Initial value 999 means not tested yet
    bool m_unrecoverable = false;
};
}
