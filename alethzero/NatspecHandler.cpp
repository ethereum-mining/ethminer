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
#include "NatspecHandler.h"
#include <boost/filesystem.hpp>
#include <string>

#include <libethereum/Defaults.h>
#include <libdevcore/Common.h>

using namespace dev;
using namespace dev::eth;

NatspecHandler::NatspecHandler()
{
	std::string path = Defaults::dbPath();
	boost::filesystem::create_directories(path);
	ldb::Options o;
	o.create_if_missing = true;
	ldb::DB::Open(o, path + "/natspec", &m_db);
}


void NatspecHandler::add(dev::h256 const& _contractHash, std::string const& _doc)
{
	bytes k = _contractHash.asBytes();
	std::string v = _doc;
	m_db->Put(m_writeOptions, ldb::Slice((char const*)k.data(), k.size()), ldb::Slice((char const*)v.data(), v.size()));
}

std::string NatspecHandler::retrieve(dev::h256 const& _contractHash) const
{
	bytes k = _contractHash.asBytes();
	std::string ret;
	m_db->Get(m_readOptions, ldb::Slice((char const*)k.data(), k.size()), &ret);
	return ret;
}



