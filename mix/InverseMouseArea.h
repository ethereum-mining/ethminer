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
/** @file InverseMouseArea.h
 * @author Yann yann@ethdev.com
 * @date 2015
 * Ethereum IDE client.
 */

#pragma once

#include <QQuickWindow>
#include <QQuickItem>

namespace dev
{
namespace mix
{

class InverseMouseArea: public QQuickItem
{
	Q_OBJECT
	Q_PROPERTY(bool active MEMBER m_active WRITE setActive)

public:
	InverseMouseArea(QQuickItem* _parent = 0): QQuickItem(_parent) {}
	~InverseMouseArea() { if (window()) { window()->removeEventFilter(this); } }
	void setActive(bool _v);

protected:
	void itemChange(ItemChange _c, const ItemChangeData& _v) override;
	bool eventFilter(QObject* _obj, QEvent *_ev) override;
	bool contains(const QPointF& _point) const override;

private:
	bool m_active;

signals:
	void clickedOutside(QPointF _point);
};

}
}

