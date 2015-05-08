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
/** @file EthashAux.cpp
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 */

#include <libethash/ethash.h>
#include "Ethash.h"

namespace dev
{
namespace eth{


class EthashAux
{
public:
	~EthashAux();

	static EthashAux* get() { if (!s_this) s_this = new EthashAux(); return s_this; }

	struct LightAllocation
	{
		LightAllocation(uint64_t _blockNumber);
		~LightAllocation();
		bytesConstRef data() const;
		Ethash::Result compute(h256 const& _headerHash, Nonce const& _nonce) const;
		ethash_light_t light;
		uint64_t size;
	};

	struct FullAllocation
	{
		FullAllocation(ethash_light_t _light, ethash_callback_t _cb);
		~FullAllocation();
		Ethash::Result compute(h256 const& _headerHash, Nonce const& _nonce) const;
		bytesConstRef data() const;
		uint64_t size() const { return ethash_full_dag_size(full); }
		ethash_full_t full;
	};

	using LightType = std::shared_ptr<LightAllocation>;
	using FullType = std::shared_ptr<FullAllocation>;

	static h256 seedHash(unsigned _number);
	static uint64_t cacheSize(BlockInfo const& _header);

	static LightType light(BlockInfo const& _header);
	static LightType light(uint64_t _blockNumber);
	static FullType full(BlockInfo const& _header);
	static FullType full(uint64_t _blockNumber);

	static Ethash::Result eval(BlockInfo const& _header) { return eval(_header, _header.nonce); }
	static Ethash::Result eval(BlockInfo const& _header, Nonce const& _nonce);
	static Ethash::Result eval(uint64_t _blockNumber, h256 const& _headerHash, Nonce const& _nonce);


private:
	EthashAux() {}

	void killCache(h256 const& _s);

	static EthashAux* s_this;
	RecursiveMutex x_this;

	std::map<h256, std::shared_ptr<LightAllocation>> m_lights;
	std::map<h256, std::weak_ptr<FullAllocation>> m_fulls;
	FullType m_lastUsedFull;

	Mutex x_epochs;
	std::map<h256, unsigned> m_epochs;
	h256s m_seedHashes;
};

}
}
