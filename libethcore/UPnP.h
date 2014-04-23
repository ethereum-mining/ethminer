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
/** @file UPnP.h
 * @authors:
 *   Gav Wood <i@gavwood.com>
 * @date 2014
 */

#pragma once

#include <set>
#include <string>

struct UPNPUrls;
struct IGDdatas;

namespace eth
{

class UPnP
{
public:
	UPnP();
	~UPnP();

	std::string externalIP();
	int addRedirect(char const* addr, int port);
	void removeRedirect(int port);

	bool isValid() const { return m_ok; }

	std::set<int> m_reg;
	bool m_ok;
	std::unique_ptr<struct UPNPUrls> m_urls;
	std::unique_ptr<struct IGDdatas> m_data;
};

}
