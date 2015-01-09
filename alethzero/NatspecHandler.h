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

/** @file NatspecHandler.h
 * @author Lefteris Karapetsas <lefteris@ethdev.com>
 * @date 2015
 */

#pragma once

#pragma warning(push)
#pragma warning(disable: 4100 4267)
#include <leveldb/db.h>
#pragma warning(pop)
#include <libdevcore/FixedHash.h>

namespace ldb = leveldb;

class NatspecHandler
{
  public:
	NatspecHandler();

	void add(dev::h256 const& _contractHash, std::string const& _doc);
	std::string retrieve(dev::h256 const& _contractHash) const;
	
  private:
	ldb::ReadOptions m_readOptions;
	ldb::WriteOptions m_writeOptions;
	ldb::DB* m_db;
};
