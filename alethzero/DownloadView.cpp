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
	unsigned syncCount = syncUnverified + bqs.unknown - syncFrom;

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

	unsigned from = min(min(hashFrom, downloadFrom), syncFrom);
	unsigned count = max(max(hashFrom + hashCount, downloadFrom + downloadCount), syncFrom + syncCount) - from;

	if (!count)
	{
		m_lastFrom = m_lastTo = (unsigned)-1;
		return;
	}

	cnote << "Range " << from << "-" << (from + count) << "(" << hashFrom << "+" << hashCount << "," << downloadFrom << "+" << downloadCount << "," << syncFrom << "+" << syncCount << ")";
	auto r = [&](unsigned u) {
		return toString((u - from) * 100 / count) + "%";
	};

	if (count)
	{
		cnote << "Hashes:" << r(hashDone) << "   Blocks:" << r(downloadFlank) << r(downloadDone) << r(downloadPoint);
		cnote << "Importing:" << r(syncFrom) << r(syncImported) << r(syncImporting) << r(syncVerified) << r(syncVerifying) << r(syncUnverified);
	}

	float squareSize = min(rect().width(), rect().height());
	QPen pen;
	pen.setCapStyle(Qt::FlatCap);
	pen.setWidthF(squareSize / 20);
	auto middle = [&](float x) {
		return QRectF(squareSize / 2 - squareSize / 2 * x, 0 + squareSize / 2 - squareSize / 2 * x, squareSize * x, squareSize * x);
	};

	auto arcLen = [&](unsigned x) {
		return x * -5760.f / count;
	};
	auto arcPos = [&](unsigned x) {
		return int(90 * 16.f + arcLen(x - from)) % 5760;
	};

	p.setPen(Qt::NoPen);
	p.setBrush(QColor::fromHsv(0, 0, 210));
	pen.setWidthF(0.f);
	p.drawPie(middle(0.4f), arcPos(from), arcLen(hashDone - from));

	auto progress = [&](unsigned h, unsigned s, unsigned v, float size, float thickness, unsigned nfrom, unsigned ncount) {
		p.setBrush(Qt::NoBrush);
		pen.setColor(QColor::fromHsv(h, s, v));
		pen.setWidthF(squareSize * thickness);
		p.setPen(pen);
		p.drawArc(middle(size), arcPos(nfrom), arcLen(ncount));
	};

	progress(0, 50, 170, 0.4f, 0.12f, downloadFlank, downloadPoint - downloadFlank);
	progress(0, 0, 150, 0.4f, 0.10f, from, downloadDone - from);

	progress(0, 0, 230, 0.7f, 0.090f, from, syncUnverified - from);
	progress(60, 25, 210, 0.7f, 0.08f, from, syncVerifying - from);
	progress(120, 25, 190, 0.7f, 0.07f, from, syncVerified - from);

	progress(0, 0, 220, 0.9f, 0.02f, from, count);
	progress(0, 0, 100, 0.9f, 0.04f, from, syncFrom - from);
	progress(0, 50, 100, 0.9f, 0.08f, syncFrom, syncImporting - syncFrom);

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
