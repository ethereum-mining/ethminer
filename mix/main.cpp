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
 * Ethereum IDE client.
 */

#include <iostream>
#include <stdlib.h>
#include "MixApplication.h"
#include "Exceptions.h"
using namespace dev::mix;

int main(int _argc, char* _argv[])
{
#ifdef ETH_HAVE_WEBENGINE
	Q_INIT_RESOURCE(js);
#endif
#if __linux
	//work around ubuntu appmenu-qt5 bug
	//https://bugs.launchpad.net/ubuntu/+source/appmenu-qt5/+bug/1323853
	putenv((char*)"QT_QPA_PLATFORMTHEME=");
	putenv((char*)"QSG_RENDER_LOOP=threaded");
#endif
	try
	{
		MixApplication app(_argc, _argv);
		return app.exec();
	}
	catch (boost::exception const& _e)
	{
		std::cerr << boost::diagnostic_information(_e);
	}
	catch (std::exception const& _e)
	{
		std::cerr << _e.what();
	}
}
