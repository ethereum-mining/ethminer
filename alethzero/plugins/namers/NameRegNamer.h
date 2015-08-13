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
/** @file NameRegNamer.h
 * @author Gav Wood <i@gavwood.com>
 * @date 2015
 */

#pragma once

#include "MainFace.h"

namespace dev
{
namespace az
{

class NameRegNamer: public QObject, public AccountNamerPlugin
{
	Q_OBJECT

public:
	NameRegNamer(MainFace* _m);
	~NameRegNamer();

private:
	void readSettings(QSettings const&) override;
	void writeSettings(QSettings&) override;

	std::string toName(Address const&) const override;
	Address toAddress(std::string const&) const override;
	Addresses knownAddresses() const override;

	void updateCache();
	void killRegistrar(Address const& _r);

	Addresses m_registrars;
	std::unordered_map<Address, unsigned> m_filters;

	mutable Addresses m_knownCache;
//	mutable std::unordered_map<Address, std::string> m_forwardCache;
//	mutable std::unordered_map<std::string, Address> m_reverseCache;
};

}
}
