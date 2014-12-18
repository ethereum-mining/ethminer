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
/** @file KeyEventManager.cpp
 * @author Yann yann@ethdev.com
 * @date 2014
 * Used as an event handler for all classes which need keyboard interactions.
 * Can be improve by adding the possibility to register to a specific key.
 */

#include <QDebug>
#include <QKeySequence>
#include "KeyEventManager.h"

void KeyEventManager::registerEvent(const QObject* _receiver, const char* _slot)
{
	QObject::connect(this, SIGNAL(onKeyPressed(int)), _receiver, _slot);
}

void KeyEventManager::unRegisterEvent(QObject* _receiver)
{
	QObject::disconnect(_receiver);
}

void KeyEventManager::keyPressed(QVariant _event)
{
	emit onKeyPressed(_event.toInt());
}
