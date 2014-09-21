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
/** @file DownloadView.cpp
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 */

#include "DownloadView.h"

#include <QtWidgets>
#include <QtCore>
#include <libethereum/DownloadMan.h>
#include "Grapher.h"

using namespace std;
using namespace dev;
using namespace dev::eth;

DownloadView::DownloadView(QWidget* _p): QWidget(_p)
{
}

void DownloadView::paintEvent(QPaintEvent*)
{
	QPainter p(this);

	p.fillRect(rect(), Qt::white);
	if (!m_man || m_man->chain().empty() || !m_man->subCount())
		return;

	double ratio = (double)rect().width() / rect().height();
	if (ratio < 1)
		ratio = 1 / ratio;
	double n = min(rect().width(), rect().height()) / ceil(sqrt(m_man->chain().size() / ratio));

//	QSizeF area(rect().width() / floor(rect().width() / n), rect().height() / floor(rect().height() / n));
	QSizeF area(n, n);
	QPointF pos(0, 0);

	auto const& bg = m_man->blocksGot();

	for (unsigned i = bg.all().first, ei = bg.all().second; i < ei; ++i)
	{
		int s = -2;
		if (bg.contains(i))
			s = -1;
		else
		{
			unsigned h = 0;
			m_man->foreachSub([&](DownloadSub const& sub)
			{
				if (sub.asked().contains(i))
					s = h;
				h++;
			});
		}
		unsigned dh = 360 / m_man->subCount();
		if (s == -2)
			p.fillRect(QRectF(QPointF(pos) + QPointF(3 * area.width() / 8, 3 * area.height() / 8), area / 4), Qt::black);
		else if (s == -1)
			p.fillRect(QRectF(QPointF(pos) + QPointF(1 * area.width() / 8, 1 * area.height() / 8), area * 3 / 4), Qt::black);
		else
			p.fillRect(QRectF(QPointF(pos) + QPointF(1 * area.width() / 8, 1 * area.height() / 8), area * 3 / 4), QColor::fromHsv(s * dh, 64, 128));

		pos.setX(pos.x() + n);
		if (pos.x() >= rect().width() - n)
			pos = QPoint(0, pos.y() + n);
	}
}
