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
