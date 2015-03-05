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

#include "Context.h"
#include <QComboBox>
#include <libethcore/Common.h>
using namespace std;
using namespace dev;
using namespace dev::eth;

NatSpecFace::~NatSpecFace()
{
}

Context::~Context()
{
}

void initUnits(QComboBox* _b)
{
	for (auto n = (unsigned)units().size(); n-- != 0; )
		_b->addItem(QString::fromStdString(units()[n].second), n);
}

vector<KeyPair> keysAsVector(QList<KeyPair> const& keys)
{
	auto list = keys.toStdList();
	return {begin(list), end(list)};
}

bool sourceIsSolidity(string const& _source)
{
	// TODO: Improve this heuristic
	return (_source.substr(0, 8) == "contract" || _source.substr(0, 5) == "//sol");
}

bool sourceIsSerpent(string const& _source)
{
	// TODO: Improve this heuristic
	return (_source.substr(0, 5) == "//ser");
}
