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
	std::ostringstream s;
	s << std::hex << "0x" << value().toStdString();
	return padded(jsToBytes(s.str()), declaration()->typeLength().first().toInt() / 8);
}

int QIntType::length()
{
	return (declaration()->typeLength().first().toInt() / 8) * 2;
}

void QIntType::decodeValue(std::string const& _rawValue)
{
	std::string unPadded = unpadLeft(_rawValue);
	int x = std::stol(unPadded, nullptr, 16);
	std::stringstream ss;
	ss << std::dec << x;
	setValue(QString::fromStdString(ss.str()));
}

/*
 * QHashType
 */
dev::bytes QHashType::encodeValue()
{
	return bytes();
}

int QHashType::length()
{
	return (declaration()->typeLength().first().toInt() / 8) * 2;
}

void QHashType::decodeValue(std::string const& _rawValue)
{
	Q_UNUSED(_rawValue);
}

/*
 * QRealType
 */
dev::bytes QRealType::encodeValue()
{

	std::ostringstream s;
	s << std::hex << "0x" << value().split("x").first().toStdString();
	bytes first = padded(jsToBytes(s.str()), declaration()->typeLength().first().toInt() / 8);
	s << std::hex << "0x" << value().split("x").last().toStdString();
	bytes second = padded(jsToBytes(s.str()), declaration()->typeLength().last().toInt() / 8);
	first.insert(first.end(), second.begin(), second.end());
	return first;
}

int QRealType::length()
{
	return (declaration()->typeLength().first().toInt() / 8) * 2 + (declaration()->typeLength().last().toInt() / 8) * 2;
}

void QRealType::decodeValue(std::string const& _rawValue)
{
	QString value;
	for (QString str: declaration()->typeLength())
	{
		std::string rawParam = _rawValue.substr(0, (str.toInt() / 8) * 2);
		std::string unPadded = unpadLeft(rawParam);
		int x = std::stol(unPadded, nullptr, 16);
		std::stringstream ss;
		ss << std::dec << x;
		value += QString::fromStdString(ss.str()) + "x";
	}
	setValue(value);
}

/*
 * QStringType
 */
dev::bytes QStringType::encodeValue()
{
	return padded(jsToBytes(value().toStdString()), declaration()->typeLength().first().toInt() / 8);
}

int QStringType::length()
{
	if (declaration()->typeLength().length() == 0)
		return value().length() + 2;
	else
		return (declaration()->typeLength().first().toInt() / 8) * 2;
}

void QStringType::decodeValue(std::string const& _rawValue)
{
	if (declaration()->typeLength().first().length() == 0)
	{
		std::string strLength = _rawValue.substr(0, 2);
		std::string strValue = _rawValue.substr(2, std::stoi(strLength));
		setValue(QString::fromStdString(strValue));
	}
	else
	{
		std::string rawParam = _rawValue.substr(0, (declaration()->typeLength().first().toInt() / 8) * 2);
		setValue(QString::fromStdString(rawParam));
	}
}

/*
 * QBoolType
 */
dev::bytes QBoolType::encodeValue()
{
	return padded(jsToBytes(value().toStdString()), 1);
}

int QBoolType::length()
{
	return 1;
}

void QBoolType::decodeValue(std::string const& _rawValue)
{
	std::string rawParam = _rawValue.substr(0, 2);
	std::string unpadded = unpadLeft(rawParam);
	setValue(QString::fromStdString(unpadded));
}

