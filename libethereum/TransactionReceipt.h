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
/** @file TransactionReceipt.h
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 */

#pragma once

#include <array>
#include <map>
#include <libdevcore/Common.h>
#include <libdevcore/RLP.h>
#include <libevm/ExtVMFace.h>

namespace dev
{

namespace eth
{

class TransactionReceipt
{
public:
	TransactionReceipt(bytesConstRef _rlp);
	TransactionReceipt(h256 _root, u256 _gasUsed, LogEntries const& _log);

	h256 const& stateRoot() const { return m_stateRoot; }
	u256 const& gasUsed() const { return m_gasUsed; }
	LogBloom const& bloom() const { return m_bloom; }
	LogEntries const& log() const { return m_log; }

	void streamRLP(RLPStream& _s) const;

	bytes rlp() const { RLPStream s; streamRLP(s); return s.out(); }

private:
	h256 m_stateRoot;
	u256 m_gasUsed;
	LogBloom m_bloom;
	LogEntries m_log;
};

using TransactionReceipts = std::vector<TransactionReceipt>;

std::ostream& operator<<(std::ostream& _out, eth::TransactionReceipt const& _r);

}
}
