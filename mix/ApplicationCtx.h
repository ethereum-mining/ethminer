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
/** @file ApplicationCtx.h
 * @author Yann yann@ethdev.com
 * @date 2014
 * Ethereum IDE client.
 */

#ifndef APPLICATIONCONTEXT_H
#define APPLICATIONCONTEXT_H

#include <QQmlApplicationEngine>

class ApplicationCtx : public QObject
{
    Q_OBJECT

public:
    ApplicationCtx(QQmlApplicationEngine*);
    ~ApplicationCtx();
    QQmlApplicationEngine* appEngine();
    static ApplicationCtx* GetInstance();
    static void SetApplicationContext(QQmlApplicationEngine*);
private:
    static ApplicationCtx* m_instance;
    QQmlApplicationEngine* m_applicationEngine;
public slots:
    void QuitApplication();
};

#endif // APPLICATIONCTX_H
