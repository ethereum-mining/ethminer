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
/** @file InverseMouseArea.cpp
 * @author Yann yann@ethdev.com
 * @date 2015
 * Ethereum IDE client.
 */

#include <QQuickWindow>
#include <QDebug>
#include <QQuickItem>
#include <QGraphicsSceneMouseEvent>
#include "InverseMouseArea.h"

using namespace dev::mix;

void InverseMouseArea::itemChange(ItemChange _c, const ItemChangeData& _v)
{
	Q_UNUSED(_v);
	if (this->m_active && _c == ItemSceneChange && window())
		window()->installEventFilter(this);
}

bool InverseMouseArea::eventFilter(QObject* _obj, QEvent* _ev)
{
	Q_UNUSED(_obj);
	if (this->m_active && _ev->type() == QEvent::MouseButtonPress && !this->contains(static_cast<QMouseEvent*>(_ev)->pos()))
		emit clickedOutside(QPointF(static_cast<QMouseEvent*>(_ev)->pos()));
	return false;
}

bool InverseMouseArea::contains(const QPointF& _point) const
{
	if (!this->m_active)
		return false;
	QPointF global = this->parentItem()->mapToItem(0, QPointF(0, 0));
	return QRectF(global.x(), global.y(), this->parentItem()->width(), this->parentItem()->height()).contains(_point);
}

void InverseMouseArea::setActive(bool _v)
{
	m_active = _v;
	if (m_active && window())
		window()->installEventFilter(this);
}
