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
/** @file QEther.cpp
 * @author Yann yann@ethdev.com
 * @date 2014
 */

#include <QMetaEnum>
#include "QEther.h"

using namespace dev::mix;

QString QEther::format() const
{
	return QString::fromStdString(dev::eth::formatBalance(boost::get<dev::u256>(toWei()->internalValue())));
}

QBigInt* QEther::toWei() const
{
	QMetaEnum units = staticMetaObject.enumerator(staticMetaObject.indexOfEnumerator("EtherUnit"));
	const char* key = units.valueToKey(m_currentUnit);
	for (std::pair<dev::u256, std::string> rawUnit: dev::eth::units())
	{
		if (QString::fromStdString(rawUnit.second).toLower() == QString(key).toLower())
			return multiply(new QBigInt(rawUnit.first));
	}
	return new QBigInt(dev::u256(0));
}

void QEther::setUnit(QString const& _unit)
{
	QMetaEnum units = staticMetaObject.enumerator(staticMetaObject.indexOfEnumerator("EtherUnit"));
	for (int k = 0; k < units.keyCount(); k++)
	{
		if (QString(units.key(k)).toLower() == _unit.toLower())
		{
			m_currentUnit = static_cast<EtherUnit>(units.keysToValue(units.key(k)));
			return;
		}
	}
}
