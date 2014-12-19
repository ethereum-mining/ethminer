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
/** @file KeyEventManager.h
 * @author Yann yann@ethdev.com
 * @date 2014
 * Used as an event handler for all classes which need keyboard interactions
 */

#pragma once

#include <QObject>

class KeyEventManager: public QObject
{
	Q_OBJECT

public:
	KeyEventManager() {}
	/// Allows _receiver to handle key pressed event.
	void registerEvent(const QObject* _receiver, const char* _slot);
	/// Unregister _receiver.
	void unRegisterEvent(QObject* _receiver);

signals:
	/// Emited when a key is pressed.
	void onKeyPressed(int _event);

public slots:
	/// Called when a key is pressed.
	void keyPressed(QVariant _event);
};

