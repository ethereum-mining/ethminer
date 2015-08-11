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
/** @file DappLoader.h
 * @author Arkadiy Paronyan <arkadiy@ethdev.org>
 * @date 2015
 */

#pragma once

#include <map>
#include <string>
#include <vector>
#include <QObject>
#include <QString>
#include <QUrl>
#include <QNetworkAccessManager>
#include <libdevcore/FixedHash.h>
#include <libdevcrypto/Common.h>

namespace dev
{

class WebThreeDirect;
class RLP;

namespace az
{

struct ManifestEntry
{
	std::string path;
	dev::h256 hash;
	std::string contentType;
	unsigned httpStatus;
};

struct Manifest
{
	std::vector<ManifestEntry> entries;
};

struct Dapp
{
	Manifest manifest;
	std::map<dev::h256, dev::bytes> content;
};


struct DappLocation
{
	QString canonDomain;
	QString path;
	QString contentUri;
	dev::h256 contentHash;
};

///Downloads, unpacks and prepares DApps for hosting
class DappLoader: public QObject
{
	Q_OBJECT
public:
	DappLoader(QObject* _parent, dev::WebThreeDirect* _web3, dev::Address _nameReg);
	///Load a new DApp. Resolves a name with a name reg contract. Asynchronous. dappReady is emitted once everything is read, dappError othervise
	///@param _uri Eth name path
	void loadDapp(QString const& _uri);
	///Load a regular html page
	///@param _uri Page Uri
	void loadPage(QString const& _uri);

	void setSessionKey(std::string const& _s) { m_sessionKey = _s; }

signals:
	void dappReady(Dapp& _dapp);
	void pageReady(QByteArray const& _content, QString const& _mimeType, QUrl const& _uri);
	void dappError();

private slots:
	void downloadComplete(QNetworkReply* _reply);

private:
	dev::WebThreeDirect* web3() const { return m_web3; }
	DappLocation resolveAppUri(QString const& _uri);
	void loadDapp(dev::RLP const& _rlp);
	Manifest loadManifest(std::string const& _manifest);
	QByteArray const& web3Content();

	dev::WebThreeDirect* m_web3;
	QNetworkAccessManager m_net;
	std::map<QUrl, dev::h256> m_uriHashes;
	std::set<QUrl> m_pageUrls;
	QByteArray m_web3Js;
	dev::Address m_nameReg;
	std::string m_sessionKey;
};

}
}
