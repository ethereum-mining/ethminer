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
/** @file DebuggingStateWrapper.h
 * @author Yann yann@ethdev.com
 * @date 2014
 * Ethereum IDE client.
 */

#pragma once

#include <QStringList>
#include <QMap>
#include <libdevcore/Common.h>
#include <libethereum/State.h>
#include <libethereum/Executive.h>
#include "QVariableDefinition.h"
#include "MixClient.h"
#include "QBigInt.h"

namespace dev
{
namespace mix
{

/**
 * @brief Contains the line nb of the assembly code and the corresponding index in the code bytes array.
 */
class HumanReadableCode: public QObject
{
	Q_OBJECT
	Q_PROPERTY(QString line READ line CONSTANT)
	Q_PROPERTY(int processIndex READ processIndex CONSTANT)

public:
	HumanReadableCode(QString _line, int _processIndex): QObject(), m_line(_line), m_processIndex(_processIndex) {}
	/// Get the assembly code line.
	QString line() { return m_line; }
	/// Get corresponding index.
	int processIndex() { return m_processIndex; }

private:
	QString m_line;
	int m_processIndex;
};


/**
 * @brief Publish QMap type to QML.
 */
class QQMLMap: public QObject
{
	Q_OBJECT

public:
	QQMLMap(QMap<int, int> _map): QObject(), m_map(_map) { }
	/// Get the value associated with _key store in n_map.
	Q_INVOKABLE int getValue(int _key) { return m_map.value(_key); }

private:
	QMap<int, int> m_map;
};

/**
 * @brief Wrap DebuggingState in QObject
 */
class DebuggingStateWrapper: public QObject
{
	Q_OBJECT
	Q_PROPERTY(int step READ step CONSTANT)
	Q_PROPERTY(int curPC READ curPC CONSTANT)
	Q_PROPERTY(QBigInt* gasCost READ gasCost CONSTANT)
	Q_PROPERTY(QBigInt* gas READ gas CONSTANT)
	Q_PROPERTY(QString instruction READ instruction CONSTANT)
	Q_PROPERTY(QStringList debugStack READ debugStack CONSTANT)
	Q_PROPERTY(QStringList debugStorage READ debugStorage CONSTANT)
	Q_PROPERTY(QVariantList debugMemory READ debugMemory CONSTANT)
	Q_PROPERTY(QVariantList debugCallData READ debugCallData CONSTANT)
	Q_PROPERTY(QString headerInfo READ headerInfo CONSTANT)
	Q_PROPERTY(QString endOfDebug READ endOfDebug CONSTANT)
	Q_PROPERTY(QBigInt* newMemSize READ newMemSize CONSTANT)
	Q_PROPERTY(QStringList levels READ levels CONSTANT)

public:
	DebuggingStateWrapper(bytes _code, bytes _data): QObject(), m_code(_code), m_data(_data) {}
	/// Get the step of this machine states.
	int step() { return  (int)m_state.steps; }
	/// Get the proccessed code index.
	int curPC() { return (int)m_state.curPC; }
	/// Get gas cost.
	QBigInt* gasCost();
	/// Get gas used.
	QBigInt* gas();
	/// Get stack.
	QStringList debugStack();
	/// Get storage.
	QStringList debugStorage();
	/// Get memory.
	QVariantList debugMemory();
	/// Get call data.
	QVariantList debugCallData();
	/// Get info to be displayed in the header.
	QString headerInfo();
	/// get end of debug information.
	QString endOfDebug();
	/// Get the new memory size.
	QBigInt* newMemSize();
	/// Get current instruction
	QString instruction();
	/// Get all previous steps.
	QStringList levels();
	/// Get the current processed machine state.
	MachineState state() { return m_state; }
	/// Set the current processed machine state.
	void setState(MachineState _state) { m_state = _state;  }
	/// Convert all machine state in human readable code.
	static std::tuple<QList<QObject*>, QQMLMap*> getHumanReadableCode(bytes const& _code);

private:
	MachineState m_state;
	bytes m_code;
	bytes m_data;
	QStringList fillList(QStringList& _list, QString const& _emptyValue);
	QVariantList fillList(QVariantList _list, QVariant const& _emptyValue);
	QVariantList qVariantDump(std::vector<std::vector<std::string>> const& _dump);
	/// Nicely renders the given bytes to a string, store the content in an array.
	/// @a _bytes: bytes array to be rendered as string. @a _width of a bytes line.
	std::vector<std::vector<std::string>> memDumpToList(bytes const& _bytes, unsigned _width);

};

}
}
