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
/** @file StateListView.cpp
 * @author Arkadiy Paronyan arkadiy@ethdev.com
 * @date 2014
 * Ethereum IDE client.
 */

#include <QQuickItem>
#include <QApplication>
#include <QQmlApplicationEngine>
#include <QQmlContext>
#include <QDebug>
#include "StateListView.h"

using namespace dev::mix;

StateListView::StateListView(AppContext* _context): Extension(_context, ExtensionDisplayBehavior::RightTab)
{
}

QString StateListView::contentUrl() const
{
	return QStringLiteral("qrc:/qml/StateList.qml");
}

QString StateListView::title() const
{
	return QApplication::tr("States");
}

void StateListView::start() const
{
}
