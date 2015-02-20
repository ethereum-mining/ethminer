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
/** @file FileIo.cpp
 * @author Arkadiy Paronyan arkadiy@ethdev.com
 * @date 2015
 * Ethereum IDE client.
 */

#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QTextStream>
#include <QUrl>
#include "FileIo.h"

using namespace dev::mix;

void FileIo::makeDir(QString const& _url)
{
	QUrl url(_url);
	QDir dirPath(url.path());
	if (!dirPath.exists())
		dirPath.mkpath(dirPath.path());
}

QString FileIo::readFile(QString const& _url)
{
	QUrl url(_url);
	QString path(url.path());
	if (url.scheme() == "qrc")
		path = ":" + path;
	QFile file(path);
	if (file.open(QIODevice::ReadOnly | QIODevice::Text))
	{
		QTextStream stream(&file);
		QString data = stream.readAll();
		return data;
	}
	else
		error(tr("Error reading file %1").arg(_url));
	return QString();
}

void FileIo::writeFile(QString const& _url, QString const& _data)
{
	QUrl url(_url);
	QFile file(url.path());
	if (file.open(QIODevice::WriteOnly | QIODevice::Text))
	{
		QTextStream stream(&file);
		stream << _data;
	}
	else
		error(tr("Error writing file %1").arg(_url));
}

void FileIo::copyFile(QString const& _sourceUrl, QString const& _destUrl)
{
	if (QUrl(_sourceUrl).scheme() == "qrc")
	{
		writeFile(_destUrl, readFile(_sourceUrl));
		return;
	}

	QUrl sourceUrl(_sourceUrl);
	QUrl destUrl(_destUrl);
	if (!QFile::copy(sourceUrl.path(), destUrl.path()))
		error(tr("Error copying file %1 to %2").arg(_sourceUrl).arg(_destUrl));
}

QString FileIo::getHomePath() const
{
	return QDir::homePath();
}

void FileIo::moveFile(QString const& _sourceUrl, QString const& _destUrl)
{
	QUrl sourceUrl(_sourceUrl);
	QUrl destUrl(_destUrl);
	if (!QFile::rename(sourceUrl.path(), destUrl.path()))
		error(tr("Error moving file %1 to %2").arg(_sourceUrl).arg(_destUrl));
}

bool FileIo::fileExists(QString const& _url)
{
	QUrl url(_url);
	QFile file(url.path());
	return file.exists();
}
