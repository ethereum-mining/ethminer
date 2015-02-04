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
/** @file MixApplication.cpp
 * @author Yann yann@ethdev.com
 * @date 2014
 */

#include <QDebug>
#include <QQmlApplicationEngine>

#ifdef ETH_HAVE_WEBENGINE
#include <QtWebEngine/QtWebEngine>
#endif

#include "MixApplication.h"
#include "AppContext.h"

#include <QMenuBar>

using namespace dev::mix;

MixApplication::MixApplication(int _argc, char* _argv[]):
	QApplication(_argc, _argv), m_engine(new QQmlApplicationEngine()), m_appContext(new AppContext(m_engine.get()))
{
	setOrganizationName(tr("Ethereum"));
	setOrganizationDomain(tr("ethereum.org"));
	setApplicationName(tr("Mix"));
	setApplicationVersion("0.1");
#ifdef ETH_HAVE_WEBENGINE
	QtWebEngine::initialize();
#endif
	QObject::connect(this, SIGNAL(lastWindowClosed()), context(), SLOT(quitApplication())); //use to kill ApplicationContext and other stuff
	m_appContext->load();
}

MixApplication::~MixApplication()
{
}
