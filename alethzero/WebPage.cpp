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
/** @file WebPage.cpp
 * @author Arkadiy Paronyan arkadiy@ethdev.com>
 * @date 2015
 */

#include "WebPage.h"
using namespace dev;
using namespace az;

void WebPage::javaScriptConsoleMessage(QWebEnginePage::JavaScriptConsoleMessageLevel _level, QString const& _message, int _lineNumber, QString const& _sourceID)
{
	QString prefix = _level == QWebEnginePage::ErrorMessageLevel ? "error" : _level == QWebEnginePage::WarningMessageLevel ? "warning" : "";
	emit consoleMessage(QString("%1(%2:%3):%4").arg(prefix).arg(_sourceID).arg(_lineNumber).arg(_message));
}
