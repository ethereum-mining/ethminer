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
/** @file Extension.h
 * @author Yann yann@ethdev.com
 * @date 2014
 * Ethereum IDE client.
 */

#pragma once

#include <QApplication>
#include <QQmlComponent>

class QQmlApplicationEngine;

namespace dev
{
namespace mix
{

class AppContext;

enum ExtensionDisplayBehavior
{
	HeaderView,
	RightView,
	ModalDialog
};


class Extension: public QObject
{
	Q_OBJECT

public:
	Extension(AppContext* _context);
	Extension(AppContext* _context, ExtensionDisplayBehavior _displayBehavior);
	/// Return the QML url of the view to display.
	virtual QString contentUrl() const { return ""; }
	/// Return the title of this extension.
	virtual QString title() const { return ""; }
	/// Initialize extension.
	virtual void start() const {}
	/// Add the view define in contentUrl() in the _view QObject.
	void addContentOn(QObject* _view);
	/// Add the view define in contentUrl() in the _view QObject (_view has to be a tab).
	void addTabOn(QObject* _view);
	/// Modify the display behavior of this extension.
	void setDisplayBehavior(ExtensionDisplayBehavior _displayBehavior) { m_displayBehavior = _displayBehavior; }
	/// Get the display behavior of thi extension.
	ExtensionDisplayBehavior getDisplayBehavior() { return m_displayBehavior; }

protected:
	QObject* m_view;
	ExtensionDisplayBehavior m_displayBehavior;
	AppContext* m_ctx;
	QQmlApplicationEngine* m_appEngine;

private:
	void init(AppContext* _context);
};

}
}
