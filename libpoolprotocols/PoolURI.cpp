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

#include <algorithm>
#include <iostream>
#include <map>
#include <sstream>

#include <cstring>

#include <libpoolprotocols/PoolURI.h>

using namespace dev;

struct SchemeAttributes
{
    ProtocolFamily family;
    SecureLevel secure;
    unsigned version;
};

static std::map<std::string, SchemeAttributes> s_schemes = {
    /*
    This schemes are kept for backwards compatibility.
    Ethminer do perform stratum autodetection
    */
    {"stratum+tcp", {ProtocolFamily::STRATUM, SecureLevel::NONE, 0}},
    {"stratum1+tcp", {ProtocolFamily::STRATUM, SecureLevel::NONE, 1}},
    {"stratum2+tcp", {ProtocolFamily::STRATUM, SecureLevel::NONE, 2}},
    {"stratum3+tcp", {ProtocolFamily::STRATUM, SecureLevel::NONE, 3}},
    {"stratum+tls", {ProtocolFamily::STRATUM, SecureLevel::TLS, 0}},
    {"stratum1+tls", {ProtocolFamily::STRATUM, SecureLevel::TLS, 1}},
    {"stratum2+tls", {ProtocolFamily::STRATUM, SecureLevel::TLS, 2}},
    {"stratum3+tls", {ProtocolFamily::STRATUM, SecureLevel::TLS, 3}},
    {"stratum+tls12", {ProtocolFamily::STRATUM, SecureLevel::TLS12, 0}},
    {"stratum1+tls12", {ProtocolFamily::STRATUM, SecureLevel::TLS12, 1}},
    {"stratum2+tls12", {ProtocolFamily::STRATUM, SecureLevel::TLS12, 2}},
    {"stratum3+tls12", {ProtocolFamily::STRATUM, SecureLevel::TLS12, 3}},
    {"stratum+ssl", {ProtocolFamily::STRATUM, SecureLevel::TLS12, 0}},
    {"stratum1+ssl", {ProtocolFamily::STRATUM, SecureLevel::TLS12, 1}},
    {"stratum2+ssl", {ProtocolFamily::STRATUM, SecureLevel::TLS12, 2}},
    {"stratum3+ssl", {ProtocolFamily::STRATUM, SecureLevel::TLS12, 3}},
    {"http", {ProtocolFamily::GETWORK, SecureLevel::NONE, 0}},
    {"getwork", {ProtocolFamily::GETWORK, SecureLevel::NONE, 0}},

    /*
    Any TCP scheme has, at the moment, only STRATUM protocol thus
    reiterating "stratum" word would be pleonastic
    Version 9 means auto-detect stratum mode
    */

    {"stratum", {ProtocolFamily::STRATUM, SecureLevel::NONE, 999}},
    {"stratums", {ProtocolFamily::STRATUM, SecureLevel::TLS, 999}},
    {"stratumss", {ProtocolFamily::STRATUM, SecureLevel::TLS12, 999}},

    /*
    The following scheme is only meant for simulation operations
    It's not meant to be used with -P arguments
    */

    {"simulation", {ProtocolFamily::SIMULATION, SecureLevel::NONE, 999}}
};

static bool url_decode(const std::string& in, std::string& out)
{
    out.clear();
    out.reserve(in.size());
    for (std::size_t i = 0; i < in.size(); ++i)
    {
        if (in[i] == '%')
        {
            if (i + 3 <= in.size())
            {
                int value = 0;
                std::istringstream is(in.substr(i + 1, 2));
                if (is >> std::hex >> value)
                {
                    out += static_cast<char>(value);
                    i += 2;
                }
                else
                {
                    return false;
                }
            }
            else
            {
                return false;
            }
        }
        else if (in[i] == '+')
        {
            out += ' ';
        }
        else
        {
            out += in[i];
        }
    }
    return true;
}

/*
  For a well designed explanation of URI parts
  refer to https://cpp-netlib.org/0.10.1/in_depth/uri.html
*/

URI::URI(std::string uri, bool _sim) : m_uri{std::move(uri)}
{

    std::regex sch_auth("^([a-zA-Z0-9\\+]{1,})\\:\\/\\/(.*)$");
    std::smatch matches;
    if (!std::regex_search(m_uri, matches, sch_auth, std::regex_constants::match_default))
        return;

    // Split scheme and authoority
    // Authority MUST be valued
    m_scheme = matches[1].str();
    boost::algorithm::to_lower(m_scheme);
    m_authority = matches[2].str();

    // Missing authority is not possible
    if (m_authority.empty())
        throw std::runtime_error("Invalid authority");

    // Simulation scheme is only allowed if specifically set
    if (!_sim && m_scheme == "simulation")
        throw std::runtime_error("Invalid scheme");

    // Check scheme is allowed
    if ((s_schemes.find(m_scheme) == s_schemes.end()))
        throw std::runtime_error("Invalid scheme");

    
    // Now let's see if authority part can be split into userinfo and "the rest"
    std::regex usr_url("^(.*)\\@(.*)$");
    if (std::regex_search(m_authority, matches, usr_url, std::regex_constants::match_default))
    {
        m_userinfo = matches[1].str();
        m_urlinfo = matches[2].str();
    }
    else
    {
        m_urlinfo = m_authority;
    }

    /*
      If m_userinfo present and valued it can be composed by either :
      - user
      - user.worker
      - user.worker:password
      - user:password

      In other words . delimits the beginning of worker and : delimits
      the beginning of password

    */
    if (!m_userinfo.empty())
    {

        // Save all parts enclosed in backticks into a dictionary
        // and replace them with tokens in the authority
        std::regex btick("`((?:[^`])*)`");
        std::map<std::string, std::string> btick_blocks;
        auto btick_blocks_begin =
            std::sregex_iterator(m_authority.begin(), m_authority.end(), btick);
        auto btick_blocks_end = std::sregex_iterator();
        int i = 0;
        for (std::sregex_iterator it = btick_blocks_begin; it != btick_blocks_end; ++it)
        {
            std::smatch match = *it;
            std::string match_str = match[1].str();
            btick_blocks["_" + std::to_string(i++)] = match[1].str();
        }
        if (btick_blocks.size())
        {
            std::map<std::string, std::string>::iterator it;
            for (it = btick_blocks.begin(); it != btick_blocks.end(); it++)
                boost::replace_all(m_userinfo, "`" + it->second + "`", "`" + it->first + "`");
        }

        std::vector<std::regex> usr_patterns;
        usr_patterns.push_back(std::regex("^(.*)\\.(.*)\\:(.*)$"));
        usr_patterns.push_back(std::regex("^(.*)\\:(.*)$"));
        usr_patterns.push_back(std::regex("^(.*)\\.(.*)$"));
        bool usrMatchFound = false;
        for (size_t i = 0; i < usr_patterns.size() && !usrMatchFound; i++)
        {
            if (std::regex_search(
                    m_userinfo, matches, usr_patterns.at(i), std::regex_constants::match_default))
            {
                usrMatchFound = true;
                switch (i)
                {
                case 0:
                    m_user = matches[1].str();
                    m_worker = matches[2].str();
                    m_password = matches[3].str();
                    break;
                case 1:
                    m_user = matches[1];
                    m_password = matches[2];
                    break;
                case 2:
                    m_user = matches[1];
                    m_worker = matches[2];
                    break;
                default:
                    break;
                }
            }
        }
        // If no matches found after this loop it means all the user
        // part is only user login
        if (!usrMatchFound)
            m_user = m_userinfo;

        // Replace all tokens with their respective values
        if (btick_blocks.size())
        {
            std::map<std::string, std::string>::iterator it;
            for (it = btick_blocks.begin(); it != btick_blocks.end(); it++)
            {
                boost::replace_all(m_userinfo, "`" + it->first + "`", it->second);
                boost::replace_all(m_user, "`" + it->first + "`", it->second);
                boost::replace_all(m_worker, "`" + it->first + "`", it->second);
                boost::replace_all(m_password, "`" + it->first + "`", it->second);
            }
        }

    }

    /*
      Let's process the url part which must contain at least a host
      an optional port and eventually a path (which may include a query
      and a fragment)
      Host can be a DNS host or an IP address.
      Thus we can have
      - host
      - host/path
      - host:port
      - host:port/path
    */
    size_t offset = m_urlinfo.find('/');
    if (offset != std::string::npos)
    {
        m_hostinfo = m_urlinfo.substr(0, offset);
        m_pathinfo = m_urlinfo.substr(offset);
    }
    else
    {
        m_hostinfo = m_urlinfo;
    }
    boost::algorithm::to_lower(m_hostinfo);  // needed to ensure we properly hit "exit" as host
    std::regex host_pattern("^(.*)\\:([0-9]{1,5})$");
    if (std::regex_search(m_hostinfo, matches, host_pattern, std::regex_constants::match_default))
    {
        m_host = matches[1].str();
        m_port = boost::lexical_cast<uint16_t>(matches[2].str());
    }
    else
    {
        m_host = m_hostinfo;
    }

    // Host info must be present and valued
    if (m_host.empty())
        throw std::runtime_error("Missing host");

    /*
      Eventually split path info into path query fragment
    */
    if (!m_pathinfo.empty())
    {
        // Url Decode Path

        std::vector<std::regex> path_patterns;
        path_patterns.push_back(std::regex("(\\/.*)\\?(.*)\\#(.*)$"));
        path_patterns.push_back(std::regex("(\\/.*)\\#(.*)$"));
        path_patterns.push_back(std::regex("(\\/.*)\\?(.*)$"));
        bool pathMatchFound = false;
        for (size_t i = 0; i < path_patterns.size() && !pathMatchFound; i++)
        {
            if (std::regex_search(
                    m_pathinfo, matches, path_patterns.at(i), std::regex_constants::match_default))
            {
                pathMatchFound = true;
                switch (i)
                {
                case 0:
                    m_path = matches[1].str();
                    m_query = matches[2].str();
                    m_fragment = matches[3].str();
                    break;
                case 1:
                    m_path = matches[1].str();
                    m_fragment = matches[2].str();
                    break;
                case 2:
                    m_path = matches[1].str();
                    m_query = matches[2].str();
                    break;
                default:
                    break;
                }
            }
            // If no matches found after this loop it means all the pathinfo
            // part is only path login
            if (!pathMatchFound)
                m_path = m_pathinfo;

        }
    }

    // Determine host type
    boost::system::error_code ec;
    boost::asio::ip::address address = boost::asio::ip::address::from_string(m_host, ec);
    if (!ec)
    {
        // This is a valid Ip Address
        if (address.is_v4())
            m_hostType = UriHostNameType::IPV4;
        if (address.is_v6())
            m_hostType = UriHostNameType::IPV6;

        m_isLoopBack = address.is_loopback();
    }
    else
    {
        // Check if valid DNS hostname
        std::regex hostNamePattern(
            "^(([a-zA-Z0-9]|[a-zA-Z0-9][a-zA-Z0-9\\-]*[a-zA-Z0-9])\\.)*([A-Za-z0-9]|[A-Za-z0-9][A-"
            "Za-z0-9\\-]*[A-Za-z0-9])$");
        if (std::regex_match(m_host, hostNamePattern))
            m_hostType = UriHostNameType::Dns;
        else
            m_hostType = UriHostNameType::Basic;
    }

    if (!m_user.empty())
        boost::replace_all(m_user, "`", "");
    if (!m_password.empty())
        boost::replace_all(m_password, "`", "");
    if (!m_worker.empty())
        boost::replace_all(m_worker, "`", "");

    // Eventually decode every encoded char
    std::string tmpStr;
    if (url_decode(m_userinfo, tmpStr))
        m_userinfo = tmpStr;
    if (url_decode(m_urlinfo, tmpStr))
        m_urlinfo = tmpStr;
    if (url_decode(m_hostinfo, tmpStr))
        m_hostinfo = tmpStr;
    if (url_decode(m_pathinfo, tmpStr))
        m_pathinfo = tmpStr;

    if (url_decode(m_path, tmpStr))
        m_path = tmpStr;
    if (url_decode(m_query, tmpStr))
        m_query = tmpStr;
    if (url_decode(m_fragment, tmpStr))
        m_fragment = tmpStr;
    if (url_decode(m_user, tmpStr))
        m_user = tmpStr;
    if (url_decode(m_password, tmpStr))
        m_password = tmpStr;
    if (url_decode(m_worker, tmpStr))
        m_worker = tmpStr;
}

ProtocolFamily URI::Family() const
{
    return s_schemes[m_scheme].family;
}

unsigned URI::Version() const
{
    return s_schemes[m_scheme].version;
}

std::string URI::UserDotWorker() const
{
    std::string _ret = m_user;
    if (!m_worker.empty())
        _ret.append("." + m_worker);
    return _ret;
}

SecureLevel URI::SecLevel() const
{
    return s_schemes[m_scheme].secure;
}

UriHostNameType URI::HostNameType() const
{
    return m_hostType;
}

bool URI::IsLoopBack() const
{
    return m_isLoopBack;
}

std::string URI::KnownSchemes(ProtocolFamily family)
{
    std::string schemes;
    for (const auto& s : s_schemes)
    {
        if ((s.second.family == family) && (s.second.version != 999))
            schemes += s.first + " ";
    }
    return schemes;
}
