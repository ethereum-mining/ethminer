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

#include <cstdlib>

#include "CommonData.h"
#include "Exceptions.h"

using namespace std;
using namespace dev;

int dev::fromHex(char _i, WhenError _throw)
{
    if (_i >= '0' && _i <= '9')
        return _i - '0';
    if (_i >= 'a' && _i <= 'f')
        return _i - 'a' + 10;
    if (_i >= 'A' && _i <= 'F')
        return _i - 'A' + 10;
    if (_throw == WhenError::Throw)
        BOOST_THROW_EXCEPTION(BadHexCharacter() << errinfo_invalidSymbol(_i));
    else
        return -1;
}

bytes dev::fromHex(std::string const& _s, WhenError _throw)
{
    unsigned s = (_s[0] == '0' && _s[1] == 'x') ? 2 : 0;
    std::vector<uint8_t> ret;
    ret.reserve((_s.size() - s + 1) / 2);

    if (_s.size() % 2)
    {
        int h = fromHex(_s[s++], WhenError::DontThrow);
        if (h != -1)
            ret.push_back(h);
        else if (_throw == WhenError::Throw)
            BOOST_THROW_EXCEPTION(BadHexCharacter());
        else
            return bytes();
    }
    for (unsigned i = s; i < _s.size(); i += 2)
    {
        int h = fromHex(_s[i], WhenError::DontThrow);
        int l = fromHex(_s[i + 1], WhenError::DontThrow);
        if (h != -1 && l != -1)
            ret.push_back((byte)(h * 16 + l));
        else if (_throw == WhenError::Throw)
            BOOST_THROW_EXCEPTION(BadHexCharacter());
        else
            return bytes();
    }
    return ret;
}

bool dev::setenv(const char name[], const char value[], bool override)
{
#if _WIN32
    if (!override && std::getenv(name) != nullptr)
        return true;

    return ::_putenv_s(name, value) == 0;
#else
    return ::setenv(name, value, override ? 1 : 0) == 0;
#endif
}

std::string dev::getTargetFromDiff(double diff, HexPrefix _prefix)
{
    using namespace boost::multiprecision;
    using BigInteger = boost::multiprecision::cpp_int;

    static BigInteger base("0x00000000ffff0000000000000000000000000000000000000000000000000000");
    BigInteger product;

    if (diff == 0)
    {
        product = BigInteger("0xffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff");
    }
    else
    {
        diff = 1 / diff;

        BigInteger idiff(diff);
        product = base * idiff;

        std::string sdiff = boost::lexical_cast<std::string>(diff);
        size_t ldiff = sdiff.length();
        size_t offset = sdiff.find(".");

        if (offset != std::string::npos)
        {
            // Number of decimal places
            size_t precision = (ldiff - 1) - offset;

            // Effective sequence of decimal places
            string decimals = sdiff.substr(offset + 1);

            // Strip leading zeroes. If a string begins with
            // 0 or 0x boost parser considers it hex
            decimals = decimals.erase(0, decimals.find_first_not_of('0'));

            // Build up the divisor as string - just in case
            // parser does some implicit conversion with 10^precision
            string decimalDivisor = "1";
            decimalDivisor.resize(precision + 1, '0');

            // This is the multiplier for the decimal part
            BigInteger multiplier(decimals);

            // This is the divisor for the decimal part
            BigInteger divisor(decimalDivisor);

            BigInteger decimalproduct;
            decimalproduct = base * multiplier;
            decimalproduct /= divisor;

            // Add the computed decimal part
            // to product
            product += decimalproduct;
        }
    }

    // Normalize to 64 chars hex with "0x" prefix
    stringstream ss;
    ss << (_prefix == HexPrefix::Add ? "0x" : "") << setw(64) << setfill('0') << std::hex
       << product;

    string target = ss.str();
    boost::algorithm::to_lower(target);
    return target;
}

double dev::getHashesToTarget(string _target)
{
    using namespace boost::multiprecision;
    using BigInteger = boost::multiprecision::cpp_int;

    static BigInteger dividend(
        "0xffff000000000000000000000000000000000000000000000000000000000000");
    BigInteger divisor(_target);
    return double(dividend / divisor);
}

std::string dev::getScaledSize(double _value, double _divisor, int _precision, string _sizes[],
    size_t _numsizes, ScaleSuffix _suffix)
{
    double _newvalue = _value;
    size_t i = 0;
    while (_newvalue > _divisor && i <= (_numsizes - 1))
    {
        _newvalue /= _divisor;
        i++;
    }

    std::stringstream _ret;
    _ret << fixed << setprecision(_precision) << _newvalue;
    if (_suffix == ScaleSuffix::Add)
        _ret << " " << _sizes[i];
    return _ret.str();
}

std::string dev::getFormattedHashes(double _hr, ScaleSuffix _suffix, int _precision)
{
    static string suffixes[] = {"h", "Kh", "Mh", "Gh", "Th", "Ph"};
    return dev::getScaledSize(_hr, 1000.0, _precision, suffixes, 6, _suffix);
}

std::string dev::getFormattedMemory(double _mem, ScaleSuffix _suffix, int _precision)
{
    static string suffixes[] = {"B", "KB", "MB", "GB", "TB", "PB"};
    return dev::getScaledSize(_mem, 1024.0, _precision, suffixes, 6, _suffix);
}

std::string dev::padLeft(std::string _value, size_t _length, char _fillChar) 
{
    if (_length > _value.size())
        _value.insert(0, (_length - _value.size()), _fillChar);
    return _value;
}

std::string dev::padRight(std::string _value, size_t _length, char _fillChar)
{
    if (_length > _value.size())
        _value.resize(_length, _fillChar);
    return _value;
}
