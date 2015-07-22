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
/** @file BasicGasPricer.h
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 */

#pragma once

#include <array>
#include "GasPricer.h"

namespace dev
{
namespace eth
{

class BasicGasPricer: public GasPricer
{
public:
	explicit BasicGasPricer(u256 _weiPerRef, u256 _refsPerBlock): m_weiPerRef(_weiPerRef), m_refsPerBlock(_refsPerBlock) {}

	void setRefPrice(u256 _weiPerRef) { if ((bigint)m_refsPerBlock * _weiPerRef > std::numeric_limits<u256>::max() ) BOOST_THROW_EXCEPTION(Overflow() << errinfo_comment("ether price * block fees is larger than 2**256-1, choose a smaller number.") ); else m_weiPerRef = _weiPerRef; }
	void setRefBlockFees(u256 _refsPerBlock) { if ((bigint)m_weiPerRef * _refsPerBlock > std::numeric_limits<u256>::max() ) BOOST_THROW_EXCEPTION(Overflow() << errinfo_comment("ether price * block fees is larger than 2**256-1, choose a smaller number.") ); else m_refsPerBlock = _refsPerBlock; }

	u256 ask(Block const&) const override { return m_weiPerRef * m_refsPerBlock / m_gasPerBlock; }
	u256 bid(TransactionPriority _p = TransactionPriority::Medium) const override { return m_octiles[(int)_p] > 0 ? m_octiles[(int)_p] : (m_weiPerRef * m_refsPerBlock / m_gasPerBlock); }

	void update(BlockChain const& _bc) override;

private:
	u256 m_weiPerRef;
	u256 m_refsPerBlock;
	u256 m_gasPerBlock = 3141592;
	std::array<u256, 9> m_octiles;
};

}
}
