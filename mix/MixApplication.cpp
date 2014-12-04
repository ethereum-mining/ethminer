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
/** @file main.cpp
 * @author Yann yann@ethdev.com
 * @date 2014
 */

#include <QDebug>
#include "MixApplication.h"
using namespace dev::mix;

MixApplication::MixApplication(int _argc, char *_argv[]): QApplication(_argc, _argv)
{
}

bool MixApplication::notify(QObject* _receiver, QEvent* _event)
{
	try
	{
		return MixApplication::notify(_receiver, _event);
	}
	catch (std::exception& _ex)
	{
		qDebug() << "std::exception was caught " << _ex.what();
	}
	catch (...)
	{
		qDebug() << "uncaught exception ";
	}
	return false;
}
