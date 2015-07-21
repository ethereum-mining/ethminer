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
/** @file EthashSealEngine.h
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 *
 * Determines the PoW algorithm.
 */

#pragma once

#include "Sealer.h"
#include "Ethash.h"
#include "EthashAux.h"

namespace dev
{
namespace eth
{

class EthashSealEngine: public SealEngineBase<Ethash>
{
	friend class Ethash;

public:
	EthashSealEngine();

	strings sealers() const override;
	void setSealer(std::string const& _sealer) override { m_sealer = _sealer; }
	void cancelGeneration() override { m_farm.stop(); }
	void generateSeal(BlockInfo const& _bi) override;
	void onSealGenerated(std::function<void(bytes const&)> const& _f) override;

private:
	bool m_opencl = false;
	eth::GenericFarm<EthashProofOfWork> m_farm;
	std::string m_sealer = "cpu";
	Ethash::BlockHeader m_sealing;
};

}
}
