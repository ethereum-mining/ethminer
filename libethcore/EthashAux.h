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

	using LightType = void const*;
	using FullType = void const*;

	static h256 seedHash(unsigned _number);
	static ethash_params params(BlockInfo const& _header);
	static ethash_params params(h256 const& _seedHash);
	static ethash_params params(unsigned _n);
	static LightType light(BlockInfo const& _header);
	static LightType light(h256 const& _header);
	static bytesConstRef full(BlockInfo const& _header, bytesRef _dest = bytesRef());
	static bytesConstRef full(h256 const& _header, bytesRef _dest = bytesRef());

	static Ethash::Result eval(BlockInfo const& _header) { return eval(_header, _header.nonce); }
	static Ethash::Result eval(BlockInfo const& _header, Nonce const& _nonce);
	static Ethash::Result eval(h256 const& _seedHash, h256 const& _headerHash, Nonce const& _nonce);

private:
	EthashAux() {}

	void killCache(h256 const& _s);

	static EthashAux* s_this;
	RecursiveMutex x_this;

	std::map<h256, LightType> m_lights;
	std::map<h256, bytesRef> m_fulls;
	std::map<h256, unsigned> m_epochs;
	h256s m_seedHashes;
};

}
}
