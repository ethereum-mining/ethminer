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
/** @file ConstantCompilation.h
 * @author Yann yann@ethdev.com
 * @date 2014
 * Ethereum IDE client.
 */

#ifndef CONSTANTCOMPILATION_H
#define CONSTANTCOMPILATION_H

#include <QTextDocument>
#include "Feature.h"

class ConstantCompilation : public Feature
{
    Q_OBJECT

public:
    ConstantCompilation(QTextDocument* doc);
    void start() override;
    QString title() override;
    QString tabUrl() override;

private:
    QTextDocument* m_editor;
    void writeOutPut(bool success, QString content);

public Q_SLOTS:
    void compile();

};

#endif // CONSTANTCOMPILATION_H
