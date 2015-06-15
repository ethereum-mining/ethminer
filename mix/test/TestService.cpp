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
#include <iostream>
#include <QUuid>
#include <QtTest/QSignalSpy>
#include <QElapsedTimer>
#include <QQuickItem>
#include <QQuickWindow>
#include <QtTest/QTest>
#include <QtTest/qtestkeyboard.h>

namespace dev
{
namespace mix
{

enum MouseAction { MousePress, MouseRelease, MouseClick, MouseDoubleClick, MouseMove };

static void mouseEvent(MouseAction _action, QWindow* _window, QObject* _item, Qt::MouseButton _button, Qt::KeyboardModifiers _stateKey, QPointF _pos, int _delay = -1)
{
	if (_delay == -1 || _delay < 30)
		_delay = 30;
	if (_delay > 0)
		QTest::qWait(_delay);

	if (_action == MouseClick)
	{
		mouseEvent(MousePress, _window, _item, _button, _stateKey, _pos);
		mouseEvent(MouseRelease, _window, _item, _button, _stateKey, _pos);
		return;
	}

	QPoint pos = _pos.toPoint();
	QQuickItem* sgitem = qobject_cast<QQuickItem*>(_item);
	if (sgitem)
		pos = sgitem->mapToScene(_pos).toPoint();

	_stateKey &= static_cast<unsigned int>(Qt::KeyboardModifierMask);

	QMouseEvent me(QEvent::User, QPoint(), Qt::LeftButton, _button, _stateKey);
	switch (_action)
	{
	case MousePress:
		me = QMouseEvent(QEvent::MouseButtonPress, pos, _window->mapToGlobal(pos), _button, _button, _stateKey);
		break;
	case MouseRelease:
		me = QMouseEvent(QEvent::MouseButtonRelease, pos, _window->mapToGlobal(pos), _button, 0, _stateKey);
		break;
	case MouseDoubleClick:
		me = QMouseEvent(QEvent::MouseButtonDblClick, pos, _window->mapToGlobal(pos), _button, _button, _stateKey);
		break;
	case MouseMove:
		// with move event the _button is NoButton, but 'buttons' holds the currently pressed buttons
		me = QMouseEvent(QEvent::MouseMove, pos, _window->mapToGlobal(pos), Qt::NoButton, _button, _stateKey);
		break;
	default:
		break;
	}
	QSpontaneKeyEvent::setSpontaneous(&me);
	if (!qApp->notify(_window, &me))
	{
		static const char* mouseActionNames[] = { "MousePress", "MouseRelease", "MouseClick", "MouseDoubleClick", "MouseMove" };
		QString warning = QString::fromLatin1("Mouse event \"%1\" not accepted by receiving window");
		QWARN(warning.arg(QString::fromLatin1(mouseActionNames[static_cast<int>(_action)])).toLatin1().data());
	}
}

bool TestService::waitForSignal(QObject* _item, QString _signalName, int _timeout)
{
	QSignalSpy spy(_item, ("2" + _signalName.toStdString()).c_str());
	QMetaObject const* mo = _item->metaObject();

	QStringList methods;

	for (int i = mo->methodOffset(); i < mo->methodCount(); ++i)
		if (mo->method(i).methodType() == QMetaMethod::Signal)
			methods << QString::fromLatin1(mo->method(i).methodSignature());

	QElapsedTimer timer;
	timer.start();

	while (!spy.size())
	{
		int remaining = _timeout - int(timer.elapsed());
		if (remaining <= 0)
			break;
		QCoreApplication::processEvents(QEventLoop::AllEvents, remaining);
		QCoreApplication::sendPostedEvents(0, QEvent::DeferredDelete);
		QTest::qSleep(10);
	}

	return spy.size();
}

bool TestService::waitForRendering(QObject* _item, int timeout)
{
	QWindow* window = eventWindow(_item);
	return waitForSignal(window, "frameSwapped()", timeout);
}

bool TestService::keyPress(QObject* _item, int _key, int _modifiers, int _delay)
{
	QWindow* window = eventWindow(_item);
	QTest::keyPress(window, Qt::Key(_key), Qt::KeyboardModifiers(_modifiers), _delay);
	return true;
}

bool TestService::keyRelease(QObject* _item, int _key, int _modifiers, int _delay)
{
	QWindow* window = eventWindow(_item);
	QTest::keyRelease(window, Qt::Key(_key), Qt::KeyboardModifiers(_modifiers), _delay);
	return true;
}

bool TestService::keyClick(QObject* _item, int _key, int _modifiers, int _delay)
{
	QWindow* window = eventWindow(_item);
	QTest::keyClick(window, Qt::Key(_key), Qt::KeyboardModifiers(_modifiers), _delay);
	return true;
}

bool TestService::keyPressChar(QObject* _item, QString const& _character, int _modifiers, int _delay)
{
	QWindow* window = eventWindow(_item);
	QTest::keyPress(window, _character[0].toLatin1(), Qt::KeyboardModifiers(_modifiers), _delay);
	return true;
}

bool TestService::keyReleaseChar(QObject* _item, QString const& _character, int _modifiers, int _delay)
{
	QWindow* window = eventWindow(_item);
	QTest::keyRelease(window, _character[0].toLatin1(), Qt::KeyboardModifiers(_modifiers), _delay);
	return true;
}

bool TestService::keyClickChar(QObject* _item, QString const& _character, int _modifiers, int _delay)
{
	QWindow* window = eventWindow(_item);
	QTest::keyClick(window, _character[0].toLatin1(), Qt::KeyboardModifiers(_modifiers), _delay);
	return true;
}

bool TestService::mouseClick(QObject* _item, qreal _x, qreal _y, int _button, int _modifiers, int _delay)
{
	QWindow* window = qobject_cast<QWindow*>(_item);
	if (!window)
		window = eventWindow(_item);
	mouseEvent(MouseClick, window, _item, Qt::MouseButton(_button), Qt::KeyboardModifiers(_modifiers), QPointF(_x, _y), _delay);
	return true;
}

void TestService::setTargetWindow(QObject* _window)
{
	QQuickWindow* window = qobject_cast<QQuickWindow*>(_window);
	if (window)
		m_targetWindow = window;
	window->requestActivate();
}

QWindow* TestService::eventWindow(QObject* _item)
{
	QQuickItem* item = qobject_cast<QQuickItem*>(_item);
	if (item && item->window())
		return item->window();

	QWindow* window = qobject_cast<QQuickWindow*>(_item);
	if (!window && _item->parent())
		window = eventWindow(_item->parent());
	if (!window)
		window = qobject_cast<QQuickWindow*>(m_targetWindow);
	if (window)
	{
		window->requestActivate();
		std::cout << window->title().toStdString();
		return window;
	}
	item = qobject_cast<QQuickItem*>(m_targetWindow);
	if (item)
		return item->window();
	return 0;
}

QString TestService::createUuid() const
{
	return QUuid::createUuid().toString();
}

}
}
