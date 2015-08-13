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
/** @file WhisperPeers.cpp
 * @author Gav Wood <i@gavwood.com>
 * @date 2015
 */

#include "WhisperPeers.h"
#include <QSettings>
#include <libethereum/Client.h>
#include <libwhisper/WhisperHost.h>
#include <libweb3jsonrpc/WebThreeStubServerBase.h>
#include "OurWebThreeStubServer.h"
#include "ui_WhisperPeers.h"
using namespace std;
using namespace dev;
using namespace az;
using namespace eth;

DEV_AZ_NOTE_PLUGIN(WhisperPeers);

WhisperPeers::WhisperPeers(MainFace* _m):
	Plugin(_m, "WhisperPeers"),
	m_ui(new Ui::WhisperPeers)
{
	dock(Qt::RightDockWidgetArea, "Active Whispers")->setWidget(new QWidget);
	m_ui->setupUi(dock()->widget());
	startTimer(1000);
}

void WhisperPeers::timerEvent(QTimerEvent*)
{
	refreshWhispers();
}

void WhisperPeers::refreshWhispers()
{
	return;
	m_ui->whispers->clear();
	for (auto const& w: whisper()->all())
	{
		shh::Envelope const& e = w.second;
		shh::Message m;
		for (pair<Public, Secret> const& i: main()->web3Server()->ids())
			if (!!(m = e.open(shh::Topics(), i.second)))
				break;
		if (!m)
			m = e.open(shh::Topics());

		QString msg;
		if (m.from())
			// Good message.
			msg = QString("{%1 -> %2} %3").arg(m.from() ? m.from().abridged().c_str() : "???").arg(m.to() ? m.to().abridged().c_str() : "*").arg(toHex(m.payload()).c_str());
		else if (m)
			// Maybe message.
			msg = QString("{%1 -> %2} %3 (?)").arg(m.from() ? m.from().abridged().c_str() : "???").arg(m.to() ? m.to().abridged().c_str() : "*").arg(toHex(m.payload()).c_str());

		time_t ex = e.expiry();
		QString t(ctime(&ex));
		t.chop(1);
		QString item = QString("[%1 - %2s] *%3 %5 %4").arg(t).arg(e.ttl()).arg(e.workProved()).arg(toString(e.topic()).c_str()).arg(msg);
		m_ui->whispers->addItem(item);
	}
}
