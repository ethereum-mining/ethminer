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
/** @file FixedClient.h
 * @author Marek Kotewicz <marek@ethdev.com>
 * @date 2015
 */

#pragma once

#include <libethereum/ClientBase.h>
#include <libethereum/BlockChain.h>
#include <libethereum/State.h>

namespace dev
{
namespace test
{

/**
 * @brief mvp implementation of ClientBase
 * Doesn't support mining interface
 */
class FixedClient: public dev::eth::ClientBase
{
public:
	FixedClient(eth::BlockChain const& _bc, eth::Block const& _block) :  m_bc(_bc), m_block(_block) {}
	virtual ~FixedClient() {}
	
	// stub
	virtual void flushTransactions() override {}
	virtual eth::BlockChain& bc() override { BOOST_THROW_EXCEPTION(InterfaceNotSupported("FixedClient::bc()")); }
	virtual eth::BlockChain const& bc() const override { return m_bc; }
	using ClientBase::asOf;
	virtual eth::Block asOf(h256 const& _h) const override;
	virtual eth::Block preMine() const override { ReadGuard l(x_stateDB); return m_block; }
	virtual eth::Block postMine() const override { ReadGuard l(x_stateDB); return m_block; }
	virtual void setBeneficiary(Address _us) override { WriteGuard l(x_stateDB); m_block.setBeneficiary(_us); }
	virtual void prepareForTransaction() override {}

private:
	eth::BlockChain const& m_bc;
	eth::Block m_block;
	mutable SharedMutex x_stateDB;			///< Lock on the state DB, effectively a lock on m_postMine.
};

}
}
