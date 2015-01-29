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
#include <libdevcore/CommonJS.h>
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
dev::bytes QIntType::encodeValue()
{
	dev::bigint i(value().toStdString());
	if (i < 0)
		i = i + dev::bigint("0xffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff") + 1;
	std::ostringstream s;
	s << std::hex << "0x" << i;
	return padded(jsToBytes(s.str()), 32);
}

void QIntType::decodeValue(std::string const& _rawValue)
{
	std::string rawParam = _rawValue.substr(0, 32 * 2);
	dev::bigint bigint = dev::bigint("0x" + rawParam);
	if (((bigint >> 32) & 1) == 1)
		bigint = bigint - dev::bigint("0xffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff") - 1;
	std::ostringstream s;
	s << std::dec << bigint;
	setValue(QString::fromStdString(s.str()));
}

/*
 * QHashType
 */
dev::bytes QHashType::encodeValue()
{
	return padded(jsToBytes("0x" + value().toStdString()), 32);
}

void QHashType::decodeValue(std::string const& _rawValue)
{
	std::string rawParam = _rawValue.substr(0, 32 * 2);
	std::string unPadded = unpadLeft(rawParam);
	setValue(QString::fromStdString(unPadded));
}

/*
 * QRealType
 */
dev::bytes QRealType::encodeValue()
{
	return bytes();
}

void QRealType::decodeValue(std::string const& _rawValue)
{
	Q_UNUSED(_rawValue);
}

/*
 * QStringType
 */
dev::bytes QStringType::encodeValue()
{
	qDebug() << QString::fromStdString(toJS(paddedRight(asBytes(value().toStdString()), 32)));
	return paddedRight(asBytes(value().toStdString()), 32);
}

void QStringType::decodeValue(std::string const& _rawValue)
{
	std::string rawParam = _rawValue.substr(0, 32 * 2);
	rawParam = unpadRight(rawParam);
	std::string res;
	res.reserve(rawParam.size() / 2);
	for (unsigned int i = 0; i < rawParam.size(); i += 2)
	{
		std::istringstream iss(rawParam.substr(i, 2));
		int temp;
		iss >> std::hex >> temp;
		res += static_cast<char>(temp);
	}
	setValue(QString::fromStdString(res));
}

/*
 * QBoolType
 */
dev::bytes QBoolType::encodeValue()
{
	return padded(jsToBytes(value().toStdString()), 32);
}

void QBoolType::decodeValue(std::string const& _rawValue)
{
	std::string rawParam = _rawValue.substr(0, 32 * 2);
	std::string unpadded = unpadLeft(rawParam);
	setValue(QString::fromStdString(unpadded));
}
