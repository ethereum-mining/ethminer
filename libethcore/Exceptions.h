#pragma once

#include <exception>
#include "CommonIO.h"
#include "CommonData.h"
#include "FixedHash.h"

namespace eth
{

class Exception: public std::exception
{
public:
	virtual std::string description() const { return typeid(*this).name(); }
	virtual char const* what() const noexcept { return typeid(*this).name(); }
};

class BadHexCharacter: public Exception {};

class RLPException: public Exception {};
class BadCast: public RLPException {};
class BadRLP: public RLPException {};
class NoNetworking: public Exception {};
class NoUPnPDevice: public Exception {};
class RootNotFound: public Exception {};

}
