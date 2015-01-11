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
/** @file FileIo.h
 * @author Arkadiy Paronyan arkadiy@ethdev.com
 * @date 2015
 * Ethereum IDE client.
 */

#pragma once

#include <QObject>

namespace dev
{
namespace mix
{

///File services for QML
class FileIo: public QObject
{
	Q_OBJECT
	Q_PROPERTY(QString homePath READ getHomePath CONSTANT)

signals:
	/// Signalled in case of IO error
	void error(QString const& _errorText);

public:
	/// Create a directory if it does not exist. Signals on failure.
	Q_INVOKABLE void makeDir(QString const& _url);
	/// Read file contents to a string. Signals on failure.
	Q_INVOKABLE QString readFile(QString const& _url);
	/// Write contents to a file. Signals on failure.
	Q_INVOKABLE void writeFile(QString const& _url, QString const& _data);
	/// Copy a file from _sourcePath to _destPath. Signals on failure.
	Q_INVOKABLE void copyFile(QString const& _sourceUrl, QString const& _destUrl);
	/// Move (rename) a file from _sourcePath to _destPath. Signals on failure.
	Q_INVOKABLE void moveFile(QString const& _sourceUrl, QString const& _destUrl);
	/// Check if file exists
	Q_INVOKABLE bool fileExists(QString const& _url);

private:
	QString getHomePath() const;
};

}
}
