/*
    This file is part of ethminer.

    ethminer is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    ethminer is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with ethminer.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "EthashAux.h"

using namespace dev;
using namespace eth;

bool EthashAux::verify(int epoch, h256 const& _headerHash, h256 const& _mixHash, uint64_t _nonce,
    h256 const& _target) noexcept
{
    auto& context = ethash::get_global_epoch_context(epoch);
    auto header = ethash::hash256_from_bytes(_headerHash.data());
    auto mix = ethash::hash256_from_bytes(_mixHash.data());
    auto target = ethash::hash256_from_bytes(_target.data());
    return ethash::verify(context, header, mix, _nonce, target);
}

bool ProgPoWAux::verify(int epoch, int block, h256 const& _headerHash, h256 const& _mixHash,
    uint64_t _nonce, h256 const& _target) noexcept
{
    auto& context = progpow::get_global_epoch_context(epoch);
    auto header = progpow::hash256_from_bytes(_headerHash.data());
    auto mix = progpow::hash256_from_bytes(_mixHash.data());
    auto target = progpow::hash256_from_bytes(_target.data());
    return progpow::verify(context, block, header, mix, _nonce, target);
}

h256 dev::eth::ProgPoWAux::hash(int epoch, int block, h256 const& _headerHash, uint64_t _nonce)
{
    auto& context = progpow::get_global_epoch_context(epoch);
    auto header = progpow::hash256_from_bytes(_headerHash.data());

    auto r = progpow::hash(context, block, header, _nonce);
    h256 res{reinterpret_cast<byte*>(r.final_hash.bytes), h256::ConstructFromPointer};
    return res;

}
