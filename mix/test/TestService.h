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
	bool waitForSignal(QObject* _item, QString _signalName, int _timeout);
	bool keyPress(int _key, int _modifiers, int _delay);
	bool keyRelease(int _key, int _modifiers, int _delay);
	bool keyClick(int _key, int _modifiers, int _delay);
	bool keyPressChar(QString const& _character, int _modifiers, int _delay);
	bool keyReleaseChar(QString const& _character, int _modifiers, int _delay);
	bool keyClickChar(QString const& _character, int _modifiers, int _delay);

private:
	QWindow* eventWindow();
	QObject* m_targetWindow;
};

}
}
