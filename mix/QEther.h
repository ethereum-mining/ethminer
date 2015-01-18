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
/** @file QEther.h
 * @author Yann yann@ethdev.com
 * @date 2014
 * Represent an Ether value in QML (mapped to u256 in c++).
 */

#pragma once

#include <QObject>
#include <libethcore/CommonEth.h>
#include "QBigInt.h"

namespace dev
{
namespace mix
{

class QEther: public QBigInt
{
	Q_OBJECT
	Q_ENUMS(EtherUnit)
	Q_PROPERTY(QString value READ value WRITE setValue NOTIFY valueChanged)
	Q_PROPERTY(QString unit READ unit WRITE setUnit NOTIFY unitChanged)

public:
	enum EtherUnit
	{
		Uether,
		Vether,
		Dether,
		Nether,
		Yether,
		Zether,
		Eether,
		Pether,
		Tether,
		Gether,
		Mether,
		grand,
		ether,
		finney,
		szabo,
		Gwei,
		Mwei,
		Kwei,
		wei
	};

	QEther(QObject* _parent = 0): QBigInt(dev::u256(0), _parent), m_currentUnit(EtherUnit::ether) {}
	QEther(dev::u256 _value, EtherUnit _unit, QObject* _parent = 0): QBigInt(_value, _parent), m_currentUnit(_unit) {}
	~QEther() {}

	Q_INVOKABLE QString format() const;
	Q_INVOKABLE QBigInt* toWei() const;
	Q_INVOKABLE void setValue(QString const& _value) { m_internalValue = dev::jsToU256(_value.toStdString()); }
	Q_INVOKABLE QString unit() const;
	Q_INVOKABLE void setUnit(QString const& _unit);
	dev::u256 toU256Wei() { return boost::get<dev::u256>(toWei()->internalValue()); }

private:
	EtherUnit m_currentUnit;

signals:
	void valueChanged();
	void unitChanged();
};

}
}

