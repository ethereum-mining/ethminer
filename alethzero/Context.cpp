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
#include <QSpinBox>
#include <libethcore/Common.h>
using namespace std;
using namespace dev;
using namespace eth;
using namespace az;

NatSpecFace::~NatSpecFace()
{
}

Context::~Context()
{
}

void dev::az::setValueUnits(QComboBox* _units, QSpinBox* _value, u256 _v)
{
	initUnits(_units);
	if (_v > 0)
	{
		_units->setCurrentIndex(0);
		while (_v > 50000 && _units->currentIndex() < (int)(units().size() - 2))
		{
			_v /= 1000;
			_units->setCurrentIndex(_units->currentIndex() + 1);
		}
	}
	else
		_units->setCurrentIndex(6);
	_value->setValue((unsigned)_v);
}

u256 dev::az::fromValueUnits(QComboBox* _units, QSpinBox* _value)
{
	return _value->value() * units()[units().size() - 1 - _units->currentIndex()].first;
}

void dev::az::initUnits(QComboBox* _b)
{
	for (auto n = (unsigned)units().size(); n-- != 0; )
		_b->addItem(QString::fromStdString(units()[n].second), n);
}

vector<KeyPair> dev::az::keysAsVector(QList<KeyPair> const& keys)
{
	auto list = keys.toStdList();
	return {begin(list), end(list)};
}

bool dev::az::sourceIsSolidity(string const& _source)
{
	// TODO: Improve this heuristic
	return (_source.substr(0, 8) == "contract" || _source.substr(0, 5) == "//sol");
}

bool dev::az::sourceIsSerpent(string const& _source)
{
	// TODO: Improve this heuristic
	return (_source.substr(0, 5) == "//ser");
}
