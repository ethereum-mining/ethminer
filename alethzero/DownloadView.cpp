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
	p.setRenderHint(QPainter::Antialiasing, true);
	p.setRenderHint(QPainter::HighQualityAntialiasing, true);

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
//	unsigned downloadFlank = downloadFrom + (sync.state == SyncState::Blocks ? overview.firstIncomplete : downloadCount);
	unsigned downloadPoint = downloadFrom + (sync.state == SyncState::Blocks ? overview.lastComplete : downloadCount);

	unsigned hashFrom = sync.state == SyncState::Hashes ? m_client->numberFromHash(PendingBlockHash) : downloadFrom;
	unsigned hashCount = sync.state == SyncState::Hashes ? sync.hashesTotal : downloadCount;
	unsigned hashDone = hashFrom + (sync.state == SyncState::Hashes ? sync.hashesReceived : hashCount);

	QString labelText = QString("PV%1").arg(sync.protocolVersion);
	QColor labelBack = QColor::fromHsv(sync.protocolVersion == 60 ? 30 : sync.protocolVersion == 61 ? 120 : 240, 15, 220);
	QColor labelFore = labelBack.darker();
	switch (sync.state)
	{
	case SyncState::Hashes:
		m_lastSyncFrom = hashFrom;
		m_lastSyncCount = hashCount;
		m_wasEstimate = sync.hashesEstimated;
		break;
	case SyncState::Blocks:
		if (m_wasEstimate)
		{
			m_lastSyncFrom = downloadFrom;
			m_lastSyncCount = downloadCount;
			m_wasEstimate = false;
		}
		break;
	case SyncState::Idle:
		if (!syncCount)
		{
			m_lastSyncFrom = (unsigned)-1;
			m_lastSyncCount = 0;
			labelBack = QColor::fromHsv(0, 0, 200);
			labelFore = Qt::white;
			labelText = "Idle";
		}
	default: break;
	}

	unsigned from = min(min(hashFrom, downloadFrom), min(syncFrom, m_lastSyncFrom));
	unsigned count = max(max(hashFrom + hashCount, downloadFrom + downloadCount), max(syncFrom + syncCount, m_lastSyncFrom + m_lastSyncCount)) - from;

/*	cnote << "Range " << from << "-" << (from + count) << "(" << hashFrom << "+" << hashCount << "," << downloadFrom << "+" << downloadCount << "," << syncFrom << "+" << syncCount << ")";
	auto r = [&](unsigned u) {
		return toString((u - from) * 100 / count) + "%";
	};

	if (count)
	{
		cnote << "Hashes:" << r(hashDone) << "   Blocks:" << r(downloadFlank) << r(downloadDone) << r(downloadPoint);
		cnote << "Importing:" << r(syncFrom) << r(syncImported) << r(syncImporting) << r(syncVerified) << r(syncVerifying) << r(syncUnverified);
	}
*/
	QPen pen;
	pen.setCapStyle(Qt::FlatCap);
	float squareSize = min(rect().width(), rect().height());
	auto middleRect = [&](float w, float h) {
		return QRectF(squareSize / 2 - w / 2, squareSize / 2 - h / 2, w, h);
	};
	auto middle = [&](float x) {
		return middleRect(squareSize * x, squareSize * x);
	};
	auto arcLen = [&](unsigned x) {
		return x * -5760.f / count;
	};
	auto arcPos = [&](unsigned x) {
		return int(90 * 16.f + arcLen(x - from)) % 5760;
	};
	auto progress = [&](unsigned h, unsigned s, unsigned v, float size, float thickness, unsigned nfrom, unsigned ncount) {
		p.setBrush(Qt::NoBrush);
		pen.setColor(QColor::fromHsv(h, s, v));
		pen.setWidthF(squareSize * thickness);
		p.setPen(pen);
		p.drawArc(middle(size), arcPos(nfrom), arcLen(ncount));
	};

	progress(0, 0, 220, 0.6f, 0.02f, from, hashDone);								// Download rail
	progress(240, 25, 170, 0.6f, 0.02f, downloadDone, downloadPoint - downloadDone);	// Latest download point
	progress(240, 50, 120, 0.6f, 0.04f, from, downloadDone - from);					// Downloaded

	progress(0, 0, 220, 0.9f, 0.02f, from, count);								// Sync rail
	progress(0, 0, 170, 0.9f, 0.02f, from, syncUnverified - from);				// Verification rail
	progress(60, 25, 170, 0.9f, 0.02f, from, syncVerifying - from);				// Verifying.
	progress(120, 25, 170, 0.9f, 0.02f, from, syncVerified - from);				// Verified.
	progress(120, 50, 120, 0.9f, 0.04f, from, syncFrom - from);					// Imported.
	progress(240, 25, 170, 0.9f, 0.04f, syncFrom, syncImporting - syncFrom);	// Importing.

	if (sync.state != SyncState::Idle || !count)
	{
		p.setBrush(QBrush(labelBack));
		p.setFont(QFont("Helvetica", 10, QFont::Bold));
		QRectF r = p.boundingRect(middle(1.f), Qt::AlignCenter, labelText);
		r.adjust(-r.width() / 8, -r.height() / 8, r.width() / 8, r.height() / 8);
		p.setPen(QPen(labelFore, r.height() / 10));
		p.drawRoundedRect(r, r.height() / 4, r.height() / 4);
		p.drawText(r, Qt::AlignCenter, labelText);
	}
}
