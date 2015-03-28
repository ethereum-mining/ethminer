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
/** @file TestService.cpp
 * @author Arkadiy Paronyan arkadiy@ethdev.com
 * @date 2015
 * Ethereum IDE client.
 */

#include "TestService.h"
#include <QtTest/QSignalSpy>
#include <QElapsedTimer>
#include <QtTest/QTest>
#include <QtTest/qtestkeyboard.h>

namespace dev
{
namespace mix
{

bool TestService::waitForSignal(QObject* _item, QString _signalName, int _timeout)
{
	QSignalSpy spy(_item,  ("2" + _signalName.toStdString()).c_str());
	QMetaObject const* mo = _item->metaObject();

	QStringList methods;

	for(int i = mo->methodOffset(); i < mo->methodCount(); ++i) {
		if (mo->method(i).methodType() == QMetaMethod::Signal) {
			methods << QString::fromLatin1(mo->method(i).methodSignature());
		}
	}

	QElapsedTimer timer;
	timer.start();

	while (!spy.size()) {
		int remaining = _timeout - int(timer.elapsed());
		if (remaining <= 0)
			break;
		QCoreApplication::processEvents(QEventLoop::AllEvents, remaining);
		QCoreApplication::sendPostedEvents(0, QEvent::DeferredDelete);
		QTest::qSleep(10);
	}

	return spy.size();
}

bool TestService::keyPress(int _key, int _modifiers, int _delay)
{
	QWindow *window = eventWindow();
	QTest::keyPress(window, Qt::Key(_key), Qt::KeyboardModifiers(_modifiers), _delay);
	return true;
}

bool TestService::keyRelease(int _key, int _modifiers, int _delay)
{
	QWindow *window = eventWindow();
	QTest::keyRelease(window, Qt::Key(_key), Qt::KeyboardModifiers(_modifiers), _delay);
	return true;
}

bool TestService::keyClick(int _key, int _modifiers, int _delay)
{
	QWindow *window = eventWindow();
	QTest::keyClick(window, Qt::Key(_key), Qt::KeyboardModifiers(_modifiers), _delay);
	return true;
}

bool TestService::keyPressChar(QString const& _character, int _modifiers, int _delay)
{
	QTEST_ASSERT(_character.length() == 1);
	QWindow *window = eventWindow();
	QTest::keyPress(window, _character[0].toLatin1(), Qt::KeyboardModifiers(_modifiers), _delay);
	return true;
}

bool TestService::keyReleaseChar(QString const& _character, int _modifiers, int _delay)
{
	QTEST_ASSERT(_character.length() == 1);
	QWindow *window = eventWindow();
	QTest::keyRelease(window, _character[0].toLatin1(), Qt::KeyboardModifiers(_modifiers), _delay);
	return true;
}

bool TestService::keyClickChar(QString const& _character, int _modifiers, int _delay)
{
	QTEST_ASSERT(_character.length() == 1);
	QWindow *window = eventWindow();
	QTest::keyClick(window, _character[0].toLatin1(), Qt::KeyboardModifiers(_modifiers), _delay);
	return true;
}

QWindow* TestService::eventWindow()
{
	return 0;
}

}
}
