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
/** @file Debugger.h
 * @author Gav Wood <i@gavwood.com>
 * @date 2015
 */

#pragma once

#include <string>
#include <vector>
#include <QString>
#include <QList>
#include <libethcore/Common.h>

class QComboBox;
class QSpinBox;

namespace dev { namespace eth { struct StateDiff; class KeyManager; } }

#define ETH_HTML_SMALL "font-size: small; "
#define ETH_HTML_MONO "font-family: Ubuntu Mono, Monospace, Lucida Console, Courier New; font-weight: bold; "
#define ETH_HTML_DIV(S) "<div style=\"" S "\">"
#define ETH_HTML_SPAN(S) "<span style=\"" S "\">"

void initUnits(QComboBox* _b);
void setValueUnits(QComboBox* _units, QSpinBox* _value, dev::u256 _v);
dev::u256 fromValueUnits(QComboBox* _units, QSpinBox* _value);

std::vector<dev::KeyPair> keysAsVector(QList<dev::KeyPair> const& _keys);

bool sourceIsSolidity(std::string const& _source);
bool sourceIsSerpent(std::string const& _source);

class NatSpecFace
{
public:
	virtual ~NatSpecFace();

	virtual void add(dev::h256 const& _contractHash, std::string const& _doc) = 0;
	virtual std::string retrieve(dev::h256 const& _contractHash) const = 0;
	virtual std::string getUserNotice(std::string const& json, const dev::bytes& _transactionData) = 0;
	virtual std::string getUserNotice(dev::h256 const& _contractHash, dev::bytes const& _transactionDacta) = 0;
};

class Context
{
public:
	virtual ~Context();

	virtual std::string pretty(dev::Address const& _a) const = 0;
	virtual std::string prettyU256(dev::u256 const& _n) const = 0;
	virtual std::pair<dev::Address, dev::bytes> fromString(std::string const& _a) const = 0;
	virtual std::string renderDiff(dev::eth::StateDiff const& _d) const = 0;
	virtual std::string render(dev::Address const& _a) const = 0;
	virtual dev::Secret retrieveSecret(dev::Address const& _a) const = 0;
	virtual dev::eth::KeyManager& keyManager() = 0;

	virtual dev::u256 gasPrice() const = 0;
};

