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
/** @file QVariableDefinition.h
 * @author Yann yann@ethdev.com
 * @date 2014
 */

#include <libdevcore/CommonData.h>
#include <libethcore/CommonJS.h>
#include "QVariableDefinition.h"

using namespace dev::mix;
int QVariableDefinitionList::rowCount(const QModelIndex& _parent) const
{
	Q_UNUSED(_parent);
	return m_def.size();
}

QVariant QVariableDefinitionList::data(const QModelIndex& _index, int _role) const
{
	if (_role != Qt::DisplayRole)
		return QVariant();

	int i = _index.row();
	if (i < 0 || i >= m_def.size())
		return QVariant(QVariant::Invalid);

	return QVariant::fromValue(m_def.at(i));
}

QHash<int, QByteArray> QVariableDefinitionList::roleNames() const
{
	QHash<int, QByteArray> roles;
	roles[Qt::DisplayRole] = "variable";
	return roles;
}

QVariableDefinition* QVariableDefinitionList::val(int _idx)
{
	if (_idx < 0 || _idx >= m_def.size())
		return nullptr;
	return m_def.at(_idx);
}

/*
 * QIntType
 */
void QIntType::setValue(dev::bigint _value)
{
	m_bigIntvalue = _value;
	std::stringstream str;
	str << std::dec << m_bigIntvalue;
	m_value = QString::fromStdString(str.str());
}

dev::bytes QIntType::encodeValue()
{
	dev::bigint i(value().toStdString());
	bytes ret(32);
	toBigEndian((u256)i, ret);
	return ret;
}

void QIntType::decodeValue(dev::bytes const& _rawValue)
{
	dev::u256 un = dev::fromBigEndian<dev::u256>(_rawValue);
	if (un >> 255)
		setValue(-s256(~un + 1));
	else
		setValue(un);
}

/*
 * QHashType
 */
dev::bytes QHashType::encodeValue()
{
	QByteArray bytesAr = value().toLocal8Bit();
	bytes r = bytes(bytesAr.begin(), bytesAr.end());
	return padded(r, 32);
}

void QHashType::decodeValue(dev::bytes const& _rawValue)
{
	std::string _ret = asString(unpadLeft(_rawValue));
	setValue(QString::fromStdString(_ret));
}

/*
 * QRealType
 */
dev::bytes QRealType::encodeValue()
{
	return bytes();
}

void QRealType::decodeValue(dev::bytes const& _rawValue)
{
	Q_UNUSED(_rawValue);
}

/*
 * QStringType
 */
dev::bytes QStringType::encodeValue()
{
	QByteArray b = value().toUtf8();
	bytes r = bytes(b.begin(), b.end());
	return paddedRight(r, 32);
}

void QStringType::decodeValue(dev::bytes const& _rawValue)
{
	setValue(QString::fromUtf8((char*)_rawValue.data()));
}

/*
 * QBoolType
 */
dev::bytes QBoolType::encodeValue()
{
	return padded(jsToBytes(value().toStdString()), 32);
}

void QBoolType::decodeValue(dev::bytes const& _rawValue)
{
	byte ret = _rawValue.at(_rawValue.size() - 1);
	bool boolRet = (ret == byte(1));
	m_boolValue = boolRet;
	m_value = m_boolValue ? "1" : "0";
}
