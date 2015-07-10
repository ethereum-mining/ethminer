/*
	This file is part of cpp-ethereum.

	cpp-ethereum is free software: you can redistribute it and/or modify
	it under the terms of the GNU General Public License as published by
	the Free Software Foundation, either version 3 of the License, or
	(at your option) any later version.

	cpp-ethereum is distributed in the hope that it will be useful,
	but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	GNU General Public License for more details.

	You should have received a copy of the GNU General Public License
	along with cpp-ethereum.  If not, see <http://www.gnu.org/licenses/>.
*/
/** @file WhisperDB.h
* @author Vladislav Gluhovsky <vlad@ethdev.com>
* @date July 2015
 */

#pragma once

#include <libdevcore/db.h>
#include <libdevcore/FixedHash.h>
#include "Common.h"

namespace dev
{
namespace shh
{

struct FailedToOpenLevelDB: virtual Exception { FailedToOpenLevelDB(std::string _message = std::string()): Exception(_message) {} };
struct FailedInsertInLevelDB: virtual Exception { FailedInsertInLevelDB(std::string _message = std::string()): Exception(_message) {} };
struct FailedLookupInLevelDB: virtual Exception { FailedLookupInLevelDB(std::string _message = std::string()): Exception(_message) {} };
struct FailedDeleteInLevelDB: virtual Exception { FailedDeleteInLevelDB(std::string _message = std::string()): Exception(_message) {} };

class WhisperDB
{
  public:
	WhisperDB();
	~WhisperDB() {}

	std::string lookup(dev::h256 const& _key) const;
	void insert(dev::h256 const& _key, std::string const& _value);
	void kill(dev::h256 const& _key);

private:
	leveldb::ReadOptions m_readOptions;
	leveldb::WriteOptions m_writeOptions;
	std::unique_ptr<leveldb::DB> m_db;
};

}
}