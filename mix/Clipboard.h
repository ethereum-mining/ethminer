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
/** @file Clipboard.h
 * @author Yann yann@ethdev.com
 * @date 2015
 */

#pragma once

#include <QObject>

namespace dev
{
namespace mix
{

/**
 * @brief Provides access to system clipboard
 */

class Clipboard: public QObject
{
	Q_OBJECT
	Q_PROPERTY(QString text READ text WRITE setText NOTIFY clipboardChanged)

public:
	Clipboard();
	/// Copy text to clipboard
	void setText(QString _text);
	/// Get text from clipboard
	QString text() const;

signals:
	void clipboardChanged();
};

}
}
