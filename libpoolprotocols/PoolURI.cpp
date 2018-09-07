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

#include <string.h>

#include <libpoolprotocols/PoolURI.h>

using namespace dev;

typedef struct
{
    ProtocolFamily family;
    SecureLevel secure;
    unsigned version;
} SchemeAttributes;

static std::map<std::string, SchemeAttributes> s_schemes = {
    /*
    This schemes are kept for backwards compatibility.
    Ethminer do perform stratum autodetection
    */
    {"stratum+tcp", {ProtocolFamily::STRATUM, SecureLevel::NONE, 0}},
    {"stratum1+tcp", {ProtocolFamily::STRATUM, SecureLevel::NONE, 1}},
    {"stratum2+tcp", {ProtocolFamily::STRATUM, SecureLevel::NONE, 2}},
    {"stratum+tls", {ProtocolFamily::STRATUM, SecureLevel::TLS, 0}},
    {"stratum1+tls", {ProtocolFamily::STRATUM, SecureLevel::TLS, 1}},
    {"stratum2+tls", {ProtocolFamily::STRATUM, SecureLevel::TLS, 2}},
    {"stratum+tls12", {ProtocolFamily::STRATUM, SecureLevel::TLS12, 0}},
    {"stratum1+tls12", {ProtocolFamily::STRATUM, SecureLevel::TLS12, 1}},
    {"stratum2+tls12", {ProtocolFamily::STRATUM, SecureLevel::TLS12, 2}},
    {"stratum+ssl", {ProtocolFamily::STRATUM, SecureLevel::TLS12, 0}},
    {"stratum1+ssl", {ProtocolFamily::STRATUM, SecureLevel::TLS12, 1}},
    {"stratum2+ssl", {ProtocolFamily::STRATUM, SecureLevel::TLS12, 2}},
    {"http", {ProtocolFamily::GETWORK, SecureLevel::NONE, 0}},
    {"getwork", {ProtocolFamily::GETWORK, SecureLevel::NONE, 0}},

    /*
    Any TCP scheme has, at the moment, only STRATUM protocol thus
    reiterating "stratum" word would be pleonastic
    Version 9 means auto-detect stratum mode
    */

    {"stratum", {ProtocolFamily::STRATUM, SecureLevel::NONE, 999}},
    {"stratums", {ProtocolFamily::STRATUM, SecureLevel::TLS, 999}},
    {"stratumss", {ProtocolFamily::STRATUM, SecureLevel::TLS12, 999}}

};

static std::string urlDecode(std::string s)
{
    std::string ret;
    unsigned i, ii;
    for (i = 0; i < s.length(); i++)
    {
        if (int(s[i]) == '%')
        {
            sscanf(s.substr(i + 1, 2).c_str(), "%x", &ii);
            ret += char(ii);
            i = i + 2;
        }
        else if (s[i] == '+')
        {
            ret += ' ';
        }
        else
        {
            ret += s[i];
        }
    }
    return ret;
}

URI::URI(const std::string uri)
{
    m_uri = uri;

    const char* curstr = m_uri.c_str();

    // <scheme> := [a-z\0-9\+\-\.]+,  convert to lower case
    // Read scheme (mandatory)
    const char* tmpstr = strchr(curstr, ':');
    if (nullptr == tmpstr)
    {
        // Not found
        return;
    }
    // Get the scheme length
    size_t len = tmpstr - curstr;
    // Copy the scheme to the string, all lowecase, can't be url encoded
    m_scheme.append(curstr, len);
    std::transform(m_scheme.begin(), m_scheme.end(), m_scheme.begin(), ::tolower);
    if (0 != std::count_if(m_scheme.begin(), m_scheme.end(), [](char c) {
            return !(isalpha(c) || isdigit(c) || ('+' == c) || ('-' == c) || ('.' == c));
        }))
    {
        return;
    }

    // Skip ':'
    tmpstr++;
    curstr = tmpstr;

    // //<user>:<password>@<host>:<port>/<url-path>
    // Any ":", "@" and "/" must be encoded.
    // Eat "//"
    if (('/' != *curstr) || ('/' != *(curstr + 1)))
    {
            return;
    }
    curstr += 2;

    // Check if the user (and password) are specified.
    bool userpass_flag = false;
    tmpstr = curstr;
    while ('\0' != *tmpstr)
    {
        if ('@' == *tmpstr)
        {
            // Username and password are specified
            userpass_flag = true;
            break;
        }
        else if ('/' == *tmpstr)
        {
            // End of <host>:<port> specification
            break;
        }
        tmpstr++;
    }

    // User and password specification
    tmpstr = curstr;
    if (userpass_flag)
    {
        // Read username
        while (('\0' != *tmpstr) && (':' != *tmpstr) && ('@' != *tmpstr))
            tmpstr++;
        len = tmpstr - curstr;
        m_username.append(curstr, len);
        m_username = urlDecode(m_username);
        // Look for password
        curstr = tmpstr;
        if (':' == *curstr)
        {
            // Skip ':'
            curstr++;
            // Read password
            tmpstr = curstr;
            while (('\0' != *tmpstr) && ('@' != *tmpstr))
                tmpstr++;
            len = tmpstr - curstr;
            m_password.append(curstr, len);
            m_password = urlDecode(m_password);
            curstr = tmpstr;
        }
        // Skip '@'
        if ('@' != *curstr)
        {
            return;
        }
        curstr++;
    }

    bool ipv6_flag = '[' == *curstr;
    // Proceed on by delimiters with reading host
    tmpstr = curstr;
    while ('\0' != *tmpstr)
    {
        if (ipv6_flag && ']' == *tmpstr)
        {
            // End of IPv6 address.
            tmpstr++;
            break;
        }
        else if (!ipv6_flag && ((':' == *tmpstr) || ('/' == *tmpstr)))
            // Port number is specified.
            break;
        tmpstr++;
    }
    len = tmpstr - curstr;
    m_host.append(curstr, len);
    m_host = urlDecode(m_host);
    curstr = tmpstr;

    // Is port number specified?
    if (':' == *curstr)
    {
        curstr++;
        // Read port number
        tmpstr = curstr;
        while (('\0' != *tmpstr) && ('/' != *tmpstr))
            tmpstr++;
        len = tmpstr - curstr;
        std::string tempstr;
        tempstr.append(curstr, len);
        std::stringstream ss(tempstr);
        ss >> m_port;
        if (ss.fail())
        {
            return;
        }
        curstr = tmpstr;
    }

    // End of the string
    if ('\0' == *curstr)
    {
        m_valid = true;
        return;
    }

    // Skip '/'
    if ('/' != *curstr)
    {
        return;
    }
    curstr++;

    // Parse path
    tmpstr = curstr;
    while (('\0' != *tmpstr) && ('#' != *tmpstr) && ('?' != *tmpstr))
        tmpstr++;
    len = tmpstr - curstr;
    if (len)
    {
        m_path = '/';
        m_path.append(curstr, len);
        m_path = urlDecode(m_path);
    }
    curstr = tmpstr;

    // Is query specified?
    if ('?' == *curstr)
    {
        // Skip '?'
        curstr++;
        // Read query
        tmpstr = curstr;
        while (('\0' != *tmpstr) && ('#' != *tmpstr))
            tmpstr++;
        len = tmpstr - curstr;
        m_query.append(curstr, len);
        m_query = urlDecode(m_query);
        curstr = tmpstr;
    }

    // Is fragment specified?
    if ('#' == *curstr)
    {
        // Skip '#'
        curstr++;
        // Read fragment
        tmpstr = curstr;
        while ('\0' != *tmpstr)
            tmpstr++;
        len = tmpstr - curstr;
        m_fragment.append(curstr, len);
        m_fragment = urlDecode(m_fragment);
    }
    m_valid = true;
}

bool URI::KnownScheme()
{
    return s_schemes.find(m_scheme) != s_schemes.end();
}

ProtocolFamily URI::Family() const
{
    return s_schemes[m_scheme].family;
}

unsigned URI::Version() const
{
    return s_schemes[m_scheme].version;
}

SecureLevel URI::SecLevel() const
{
    return s_schemes[m_scheme].secure;
}

std::string URI::KnownSchemes(ProtocolFamily family)
{
    std::string schemes;
    for (const auto& s : s_schemes)
        if ((s.second.family == family) && (s.second.version != 999))
            schemes += s.first + " ";
    return schemes;
}

