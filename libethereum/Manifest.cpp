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
/** @file Manifest.cpp
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 */

#include "Manifest.h"
using namespace std;
using namespace eth;

Manifest::Manifest(bytesConstRef _r)
{
	RLP r(_r);
	from = r[0].toHash<Address>();
	to = r[1].toHash<Address>();
	value = r[2].toInt<u256>();
	altered = r[3].toVector<u256>();
	input = r[4].toBytes();
	output = r[5].toBytes();
	for (auto const& i: r[6])
		internal.emplace_back(i.data());
}

void Manifest::streamOut(RLPStream& _s) const
{
	_s.appendList(7) << from << to << value << altered << input << output;
	_s.appendList(internal.size());
	for (auto const& i: internal)
		i.streamOut(_s);
}
