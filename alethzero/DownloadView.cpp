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
	QPainter painter(this);
	painter.fillRect(rect(), Qt::white);
	painter.setRenderHint(QPainter::Antialiasing, true);
	painter.setRenderHint(QPainter::HighQualityAntialiasing, true);

	if (!m_client || !isVisible() || !rect().width() || !rect().height())
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
	unsigned downloadFrom = sync.state == SyncState::Idle ? m_lastSyncFrom : m_client->numberFromHash(m_client->isKnown(man->firstBlock()) ? man->firstBlock() : PendingBlockHash);
	unsigned downloadCount = sync.state == SyncState::Idle ? m_lastSyncCount : sync.blocksTotal;
	unsigned downloadDone = downloadFrom + (sync.state == SyncState::Idle ? m_lastSyncCount : sync.blocksReceived);
	unsigned downloadPoint = downloadFrom + (sync.state == SyncState::Idle ? m_lastSyncCount : man->overview().lastComplete);

	unsigned hashFrom = sync.state == SyncState::Hashes ? m_client->numberFromHash(PendingBlockHash) : downloadFrom;
	unsigned hashCount = sync.state == SyncState::Hashes ? sync.hashesTotal : downloadCount;
	unsigned hashDone = hashFrom + (sync.state == SyncState::Hashes ? sync.hashesReceived : hashCount);

	QString labelText = QString("PV%1").arg(sync.protocolVersion);
	QColor labelBack = QColor::fromHsv(sync.protocolVersion == 60 ? 30 : sync.protocolVersion == 61 ? 120 : 240, 25, 200);
	QColor labelFore = labelBack.darker();
	switch (sync.state)
	{
	case SyncState::Hashes:
		if (!syncCount || !sync.hashesEstimated)
		{
			m_lastSyncFrom = min(hashFrom, m_lastSyncFrom);
			m_lastSyncCount = max(hashFrom + hashCount, m_lastSyncFrom + m_lastSyncCount) - m_lastSyncFrom;
			m_wasEstimate = sync.hashesEstimated;
		}
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
	float const squareSize = min(rect().width(), rect().height());
	auto middleRect = [&](float w, float h) {
		return QRectF(rect().width() / 2 - w / 2, rect().height() / 2 - h / 2, w, h);
	};
	auto middle = [&](float x) {
		return middleRect(squareSize * x, squareSize * x);
	};
	auto pieProgress = [&](unsigned h, unsigned s, unsigned v, float row, float thickness, unsigned nfrom, unsigned ncount) {
		auto arcLen = [&](unsigned x) {
			return x * -5760.f / count;
		};
		auto arcPos = [&](unsigned x) {
			return int(90 * 16.f + arcLen(x - from)) % 5760;
		};
		painter.setPen(QPen(QColor::fromHsv(h, s, v), squareSize * thickness, Qt::SolidLine, Qt::FlatCap));
		painter.setBrush(Qt::NoBrush);
		painter.drawArc(middle(0.5 + row / 2), arcPos(nfrom), arcLen(ncount));
	};
	auto pieProgress2 = [&](unsigned h, unsigned s, unsigned v, float row, float orbit, float thickness, unsigned nfrom, unsigned ncount) {
		pieProgress(h, s, v, row - orbit, thickness, nfrom, ncount);
		pieProgress(h, s, v, row + orbit, thickness, nfrom, ncount);
	};
	auto pieLabel = [&](QString text, float points, QColor fore, QColor back) {
		painter.setBrush(QBrush(back));
		painter.setFont(QFont("Helvetica", points, QFont::Bold));
		QRectF r = painter.boundingRect(middle(1.f), Qt::AlignCenter, text);
		r.adjust(-r.width() / 4, -r.height() / 8, r.width() / 4, r.height() / 8);
		painter.setPen(QPen(fore, r.height() / 20));
		painter.drawRoundedRect(r, r.height() / 4, r.height() / 4);
		painter.drawText(r, Qt::AlignCenter, text);
	};

	float lineHeight = painter.boundingRect(rect(), Qt::AlignTop | Qt::AlignHCenter, "Ay").height();
	auto hProgress = [&](unsigned h, unsigned s, unsigned v, float row, float thickness, unsigned nfrom, unsigned ncount) {
		QRectF r = rect();
		painter.setPen(QPen(QColor::fromHsv(h, s, v), r.height() * thickness * 3, Qt::SolidLine, Qt::FlatCap));
		painter.setBrush(Qt::NoBrush);
		auto y = row * (r.height() - lineHeight) + lineHeight;
		painter.drawLine(QPointF((nfrom - from) * r.width() / count, y), QPointF((nfrom + ncount - from) * r.width() / count, y));
	};
	auto hProgress2 = [&](unsigned h, unsigned s, unsigned v, float row, float orbit, float thickness, unsigned nfrom, unsigned ncount) {
		hProgress(h, s, v, row - orbit * 3, thickness, nfrom, ncount);
		hProgress(h, s, v, row + orbit * 3, thickness, nfrom, ncount);
	};
	auto hLabel = [&](QString text, float points, QColor fore, QColor back) {
		painter.setBrush(QBrush(back));
		painter.setFont(QFont("Helvetica", points, QFont::Bold));
		QRectF r = painter.boundingRect(rect(), Qt::AlignTop | Qt::AlignHCenter, text);
		r.adjust(-r.width() / 4, r.height() / 8, r.width() / 4, 3 * r.height() / 8);
		painter.setPen(QPen(fore, r.height() / 20));
		painter.drawRoundedRect(r, r.height() / 4, r.height() / 4);
		painter.drawText(r, Qt::AlignCenter, text);
	};

	function<void(unsigned h, unsigned s, unsigned v, float row, float thickness, unsigned nfrom, unsigned ncount)> progress;
	function<void(unsigned h, unsigned s, unsigned v, float row, float orbit, float thickness, unsigned nfrom, unsigned ncount)> progress2;
	function<void(QString text, float points, QColor fore, QColor back)> label;
	if (rect().width() / rect().height() > 5)
	{
		progress = hProgress;
		progress2 = hProgress2;
		label = hLabel;
	}
	else
	{
		progress = pieProgress;
		progress2 = pieProgress2;
		label = pieLabel;
	}

	if (sync.state != SyncState::Idle)
	{
		progress(0, 0, 220, 0.4f, 0.02f, from, hashDone - from);							// Download rail
		progress(240, 25, 170, 0.4f, 0.02f, downloadDone, downloadPoint - downloadDone);	// Latest download point
		progress(240, 50, 120, 0.4f, 0.04f, from, downloadDone - from);						// Downloaded
	}

	progress(0, 0, 220, 0.8f, 0.01f, from, count);								// Sync rail
	progress(0, 0, 170, 0.8f, 0.02f, from, syncUnverified - from);				// Verification rail
	progress2(60, 25, 170, 0.8f, 0.06f, 0.005f, from, syncVerifying - from);	// Verifying.
	progress2(120, 25, 170, 0.8f, 0.06f, 0.005f, from, syncVerified - from);	// Verified.
	progress(120, 50, 120, 0.8f, 0.05f, from, syncFrom - from);					// Imported.
	progress(0, 0, 120, 0.8f, 0.02f, syncFrom, syncImporting - syncFrom);		// Importing.

	if (sync.state != SyncState::Idle || (sync.state == SyncState::Idle && !syncCount))
		label(labelText, 11, labelFore, labelBack);
}
