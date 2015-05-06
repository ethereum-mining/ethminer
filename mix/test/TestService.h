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
/** @file TestService.h
 * @author Arkadiy Paronyan arkadiy@ethdev.com
 * @date 2015
 * Ethereum IDE client.
 */

#pragma once

#include <QObject>

class QWindow;

namespace dev
{
namespace mix
{

class TestService: public QObject
{
	Q_OBJECT
	Q_PROPERTY(QObject* targetWindow READ targetWindow WRITE setTargetWindow)

public:
	QObject* targetWindow() const { return m_targetWindow; }
	void setTargetWindow(QObject* _window);

public slots:
	QString createUuid() const;
	bool waitForSignal(QObject* _item, QString _signalName, int _timeout);
	bool waitForRendering(QObject* _item, int timeout);
	bool keyPress(QObject* _item, int _key, int _modifiers, int _delay);
	bool keyRelease(QObject* _item, int _key, int _modifiers, int _delay);
	bool keyClick(QObject* _item, int _key, int _modifiers, int _delay);
	bool keyPressChar(QObject* _item, QString const& _character, int _modifiers, int _delay);
	bool keyReleaseChar(QObject* _item, QString const& _character, int _modifiers, int _delay);
	bool keyClickChar(QObject* _item, QString const& _character, int _modifiers, int _delay);
	bool mouseClick(QObject* _item, qreal _x, qreal _y, int _button, int _modifiers, int _delay);

private:
	QWindow* eventWindow(QObject* _item);
	QObject* m_targetWindow;
};

}
}
