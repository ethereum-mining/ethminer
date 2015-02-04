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
