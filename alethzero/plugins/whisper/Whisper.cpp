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
/** @file Whisper.cpp
 * @author Gav Wood <i@gavwood.com>
 * @date 2015
 */

#include "Whisper.h"
#include <QSettings>
#include <libethereum/Client.h>
#include <libethereum/Utility.h>
#include <libwhisper/WhisperHost.h>
#include <libweb3jsonrpc/WebThreeStubServerBase.h>
#include "OurWebThreeStubServer.h"
#include "ui_Whisper.h"
using namespace std;
using namespace dev;
using namespace az;
using namespace eth;

DEV_AZ_NOTE_PLUGIN(Whisper);

static Public stringToPublic(QString const& _a)
{
	string sn = _a.toStdString();
	if (_a.size() == sizeof(Public) * 2)
		return Public(fromHex(_a.toStdString()));
	else if (_a.size() == sizeof(Public) * 2 + 2 && _a.startsWith("0x"))
		return Public(fromHex(_a.mid(2).toStdString()));
	else
		return Public();
}

static shh::Topics topicFromText(QString _s)
{
	shh::BuildTopic ret;
	while (_s.size())
	{
		QRegExp r("(@|\\$)?\"([^\"]*)\"(\\s.*)?");
		QRegExp d("(@|\\$)?([0-9]+)(\\s*(ether)|(finney)|(szabo))?(\\s.*)?");
		QRegExp h("(@|\\$)?(0x)?(([a-fA-F0-9])+)(\\s.*)?");
		bytes part;
		if (r.exactMatch(_s))
		{
			for (auto i: r.cap(2))
				part.push_back((byte)i.toLatin1());
			if (r.cap(1) != "$")
				for (int i = r.cap(2).size(); i < 32; ++i)
					part.push_back(0);
			else
				part.push_back(0);
			_s = r.cap(3);
		}
		else if (d.exactMatch(_s))
		{
			u256 v(d.cap(2).toStdString());
			if (d.cap(6) == "szabo")
				v *= szabo;
			else if (d.cap(5) == "finney")
				v *= finney;
			else if (d.cap(4) == "ether")
				v *= ether;
			bytes bs = dev::toCompactBigEndian(v);
			if (d.cap(1) != "$")
				for (auto i = bs.size(); i < 32; ++i)
					part.push_back(0);
			for (auto b: bs)
				part.push_back(b);
			_s = d.cap(7);
		}
		else if (h.exactMatch(_s))
		{
			bytes bs = fromHex((((h.cap(3).size() & 1) ? "0" : "") + h.cap(3)).toStdString());
			if (h.cap(1) != "$")
				for (auto i = bs.size(); i < 32; ++i)
					part.push_back(0);
			for (auto b: bs)
				part.push_back(b);
			_s = h.cap(5);
		}
		else
			_s = _s.mid(1);
		ret.shift(part);
	}
	return ret;
}


Whisper::Whisper(MainFace* _m):
	Plugin(_m, "Whisper"),
	m_ui(new Ui::Whisper)
{
	dock(Qt::RightDockWidgetArea, "Whisper")->setWidget(new QWidget);
	m_ui->setupUi(dock()->widget());
	connect(addMenuItem("New Whisper identity.", "menuNetwork", true), &QAction::triggered, this, &Whisper::on_newIdentity_triggered);
	connect(_m->web3Server(), &OurWebThreeStubServer::onNewId, this, &Whisper::addNewId);
}

void Whisper::readSettings(QSettings const& _s)
{
	m_myIdentities.clear();
	QByteArray b = _s.value("identities").toByteArray();
	if (!b.isEmpty())
	{
		Secret k;
		for (unsigned i = 0; i < b.size() / sizeof(Secret); ++i)
		{
			memcpy(k.writable().data(), b.data() + i * sizeof(Secret), sizeof(Secret));
			if (!count(m_myIdentities.begin(), m_myIdentities.end(), KeyPair(k)))
				m_myIdentities.append(KeyPair(k));
		}
	}
	main()->web3Server()->setIdentities(keysAsVector(m_myIdentities));
}

void Whisper::writeSettings(QSettings& _s)
{
	QByteArray b;
	b.resize(sizeof(Secret) * m_myIdentities.size());
	auto p = b.data();
	for (auto i: m_myIdentities)
	{
		memcpy(p, &(i.secret()), sizeof(Secret));
		p += sizeof(Secret);
	}
	_s.setValue("identities", b);
}

void Whisper::addNewId(QString _ids)
{
	KeyPair kp(jsToSecret(_ids.toStdString()));
	m_myIdentities.push_back(kp);
	main()->web3Server()->setIdentities(keysAsVector(m_myIdentities));
	refreshWhisper();
}

void Whisper::refreshWhisper()
{
	m_ui->shhFrom->clear();
	for (auto i: main()->web3Server()->ids())
		m_ui->shhFrom->addItem(QString::fromStdString(toHex(i.first.ref())));
}

void Whisper::on_newIdentity_triggered()
{
	KeyPair kp = KeyPair::create();
	m_myIdentities.append(kp);
	main()->web3Server()->setIdentities(keysAsVector(m_myIdentities));
	refreshWhisper();
}

void Whisper::on_post_clicked()
{
	return;
	shh::Message m;
	m.setTo(stringToPublic(m_ui->shhTo->currentText()));
	m.setPayload(parseData(m_ui->shhData->toPlainText().toStdString()));
	Public f = stringToPublic(m_ui->shhFrom->currentText());
	Secret from;
	if (main()->web3Server()->ids().count(f))
		from = main()->web3Server()->ids().at(f);
	whisper()->inject(m.seal(from, topicFromText(m_ui->shhTopic->toPlainText()), m_ui->shhTtl->value(), m_ui->shhWork->value()));
}
