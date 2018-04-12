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

#include "EthashAux.h"
#include <libethash/internal.h>

#include <ethash/ethash.hpp>

using namespace std;
using namespace chrono;
using namespace dev;
using namespace eth;

EthashAux& EthashAux::get()
{
	static EthashAux instance;
	return instance;
}

EthashAux::LightType EthashAux::light(int epoch)
{
    EthashAux& ethash = EthashAux::get();

    Guard l(ethash.x_lights);

    auto it = ethash.m_lights.find(epoch);
    if (it != ethash.m_lights.end())
        return it->second;

    return (ethash.m_lights[epoch] = make_shared<LightAllocation>(epoch));
}

EthashAux::LightAllocation::LightAllocation(int epoch)
{
    int blockNumber = epoch * ETHASH_EPOCH_LENGTH;
    light = ethash_light_new(blockNumber);
    size = ethash_get_cachesize(blockNumber);
}

EthashAux::LightAllocation::~LightAllocation()
{
	ethash_light_delete(light);
}

bytesConstRef EthashAux::LightAllocation::data() const
{
	return bytesConstRef((byte const*)light->cache, size);
}

Result EthashAux::LightAllocation::compute(h256 const& _headerHash, uint64_t _nonce) const
{
	ethash_return_value r = ethash_light_compute(light, *(ethash_h256_t*)_headerHash.data(), _nonce);
	if (!r.success)
		BOOST_THROW_EXCEPTION(DAGCreationFailure());
	return Result{h256((uint8_t*)&r.result, h256::ConstructFromPointer), h256((uint8_t*)&r.mix_hash, h256::ConstructFromPointer)};
}

Result EthashAux::eval(int epoch, h256 const& _headerHash, uint64_t _nonce) noexcept
{
    auto headerHash = ethash::hash256::from_bytes(_headerHash.data());
    auto result = ethash::managed::hash(epoch, headerHash, _nonce);
    h256 mix{reinterpret_cast<byte*>(result.mix_hash.bytes), h256::ConstructFromPointer};
    h256 final{reinterpret_cast<byte*>(result.final_hash.bytes), h256::ConstructFromPointer};
    return {final, mix};
}