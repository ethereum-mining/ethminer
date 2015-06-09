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
/** @file QBigInt.cpp
 * @author Yann yann@ethdev.com
 * @date 2015
 */

#include <boost/variant/multivisitors.hpp>
#include <boost/variant.hpp>
#include <libethcore/CommonJS.h>
#include "QBigInt.h"

using namespace dev;
using namespace dev::mix;

QString QBigInt::value() const
{
	std::ostringstream s;
	s << m_internalValue;
	return QString::fromStdString(s.str());
}

QBigInt* QBigInt::subtract(QBigInt* const& _value) const
{
	BigIntVariant toSubtract = _value->internalValue();
	return new QBigInt(boost::apply_visitor(mix::subtract(), m_internalValue, toSubtract));
}

QBigInt* QBigInt::add(QBigInt* const& _value) const
{
	BigIntVariant toAdd = _value->internalValue();
	return new QBigInt(boost::apply_visitor(mix::add(), m_internalValue, toAdd));
}

QBigInt* QBigInt::multiply(QBigInt* const& _value) const
{
	BigIntVariant toMultiply = _value->internalValue();
	return new QBigInt(boost::apply_visitor(mix::multiply(), m_internalValue, toMultiply));
}

QBigInt* QBigInt::divide(QBigInt* const& _value) const
{
	BigIntVariant toDivide = _value->internalValue();
	return new QBigInt(boost::apply_visitor(mix::divide(), m_internalValue, toDivide));
}

QVariantMap QBigInt::checkAgainst(QString const& _type) const
{
	QVariantMap ret;
	QString type = _type;
	QString capacity = type.replace("uint", "").replace("int", "");
	if (capacity.isEmpty())
		capacity = "256";
	bigint range = 1;
	for (int k = 0; k < capacity.toInt() / 8; ++k)
		range = range * 256;
	bigint value = boost::get<bigint>(this->internalValue());
	ret.insert("valid", true);
	if (_type.startsWith("uint") && value > range - 1)
	{
		ret.insert("minValue", "0");
		std::ostringstream s;
		s << range - 1;
		ret.insert("maxValue", QString::fromStdString(s.str()));
		if (value > range)
			ret["valid"] = false;
	}
	else if (_type.startsWith("int"))
	{
		range = range / 2;
		std::ostringstream s;
		s << -range;
		ret.insert("minValue", QString::fromStdString(s.str()));
		s.str("");
		s.clear();
		s << range - 1;
		ret.insert("maxValue", QString::fromStdString(s.str()));
		if (-range > value || value > range - 1)
			ret["valid"] = false;
	}
	return ret;
}
