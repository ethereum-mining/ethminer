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

SyncView::SyncView(QWidget* _p): QWidget(_p)
{
}

void SyncView::paintEvent(QPaintEvent*)
{
	QPainter p(this);
	p.fillRect(rect(), Qt::white);

	if (!m_client)
		return;

	DownloadMan const* man = m_client->downloadMan();
	BlockQueueStatus bqs = m_client->blockQueueStatus();
	SyncStatus sync = m_client->syncStatus();

	unsigned syncFrom = m_client->numberFromHash(PendingBlockHash);
	unsigned syncImported = syncFrom;
	unsigned syncImporting = syncImported + bqs.importing;
	unsigned syncVerified = syncImporting + bqs.verified;
	unsigned syncVerifying = syncVerified + bqs.verifying;
	unsigned syncUnverified = syncVerifying + bqs.unverified;
	unsigned syncUnknown = syncUnverified + bqs.unknown;

	// best effort guess. assumes there's no forks.
	unsigned downloadFrom = m_client->numberFromHash(m_client->isKnown(man->firstBlock()) ? man->firstBlock() : PendingBlockHash);
	unsigned downloadCount = sync.blocksTotal;
	DownloadMan::Overview overview = man->overview();
	unsigned downloadDone = downloadFrom + overview.total;
	unsigned downloadFlank = downloadFrom + overview.firstIncomplete;
	unsigned downloadPoint = downloadFrom + overview.lastComplete;

	unsigned hashFrom = sync.state == SyncState::Hashes ? m_client->numberFromHash(PendingBlockHash) : downloadFrom;
	unsigned hashCount = sync.state == SyncState::Hashes ? sync.hashesTotal : downloadCount;
	unsigned hashDone = hashFrom + (sync.state == SyncState::Hashes ? sync.hashesReceived : hashCount);

	m_lastFrom = min(syncFrom, m_lastFrom);
	unsigned from = min(min(hashFrom, downloadFrom), min(syncFrom, m_lastFrom));
	unsigned count = max(hashFrom + hashCount, downloadFrom + downloadCount) - from;
	m_lastFrom = (m_lastFrom * 95 + syncFrom) / 100;

	if (!count)
	{
		m_lastFrom = (unsigned)-1;
		return;
	}

	cnote << "Range " << from << "-" << (from + count);
	auto r = [&](unsigned u) {
		return toString((u - from) * 100 / count) + "%";
	};

	if (count)
	{
		cnote << "Hashes:" << r(hashDone) << "   Blocks:" << r(downloadFlank) << r(downloadDone) << r(downloadPoint);
		cnote << "Importing:" << r(syncFrom) << r(syncImported) << r(syncImporting) << r(syncVerified) << r(syncVerifying) << r(syncUnverified) << r(syncUnknown);
	}

	if (!man || man->chainEmpty() || !man->subCount())
		return;

	float s = min(rect().width(), rect().height());
	QPen pen;
	pen.setCapStyle(Qt::FlatCap);
	pen.setWidthF(s / 10);
	p.setPen(pen);
	auto middle = [&](float x) {
		return QRectF(s / 2 - s / 2 * x, 0 + s / 2 - s / 2 * x, s * x, s * x);
	};

	auto toArc = [&](unsigned x) {
		return (x - from) * -5760.f / count;
	};
	const float arcFrom = 90 * 16.f;
	p.drawArc(middle(0.5f), arcFrom, toArc(downloadDone));
	p.drawPie(middle(0.2f), arcFrom, toArc(hashDone));
	return;

	double ratio = (double)rect().width() / rect().height();
	if (ratio < 1)
		ratio = 1 / ratio;
	double n = min(16.0, min(rect().width(), rect().height()) / ceil(sqrt(man->chainSize() / ratio)));

//	QSizeF area(rect().width() / floor(rect().width() / n), rect().height() / floor(rect().height() / n));
	QSizeF area(n, n);
	QPointF pos(0, 0);

	auto bg = man->blocksGot();
	unsigned subCount = man->subCount();
	if (subCount == 0)
		return;
	unsigned dh = 360 / subCount;
	for (unsigned i = bg.all().first, ei = bg.all().second; i < ei; ++i)
	{
		int s = -2;
		if (bg.contains(i))
			s = -1;
		else
		{
			unsigned h = 0;
			man->foreachSub([&](DownloadSub const& sub)
			{
				if (sub.askedContains(i))
					s = h;
				h++;
			});
		}
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
