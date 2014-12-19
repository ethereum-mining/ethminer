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
#include <libdevcore/Common.h>
#include <libethereum/State.h>
#include <libethereum/Executive.h>
#include "QVariableDefinition.h"

namespace dev
{
namespace mix
{

/**
 * @brief Store information about a machine state.
 */
struct DebuggingState
{
	uint64_t steps;
	dev::Address cur;
	dev::u256 curPC;
	dev::eth::Instruction inst;
	dev::bigint newMemSize;
	dev::u256 gas;
	dev::u256s stack;
	dev::bytes memory;
	dev::bigint gasCost;
	std::map<dev::u256, dev::u256> storage;
	std::vector<DebuggingState const*> levels;
};

/**
 * @brief Store information about a machine states.
 */
struct DebuggingContent
{
	QList<DebuggingState> machineStates;
	bytes executionCode;
	bytesConstRef executionData;
	Address contractAddress;
	bool contentAvailable;
	QString message;
	bytes returnValue;
	QList<QVariableDefinition*> returnParameters;
};

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
	/// get the assembly code line.
	QString line() { return m_line; }
	/// get corresponding index.
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
	/// get the value associated with _key store in n_map.
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
	Q_PROPERTY(QString gasCost READ gasCost CONSTANT)
	Q_PROPERTY(QString gas READ gas CONSTANT)
	Q_PROPERTY(QString gasLeft READ gasLeft CONSTANT)
	Q_PROPERTY(QString debugStack READ debugStack CONSTANT)
	Q_PROPERTY(QString debugStorage READ debugStorage CONSTANT)
	Q_PROPERTY(QString debugMemory READ debugMemory CONSTANT)
	Q_PROPERTY(QString debugCallData READ debugCallData CONSTANT)
	Q_PROPERTY(QString headerInfo READ headerInfo CONSTANT)
	Q_PROPERTY(QString endOfDebug READ endOfDebug CONSTANT)
	Q_PROPERTY(QStringList levels READ levels CONSTANT)

public:
	DebuggingStateWrapper(bytes _code, bytes _data): QObject(), m_code(_code), m_data(_data) {}
	/// get the step of this machine states.
	int step() { return  (int)m_state.steps; }
	/// get the proccessed code.
	int curPC() { return (int)m_state.curPC; }
	/// get gas left.
	QString gasLeft();
	/// get gas cost.
	QString gasCost();
	/// get gas used.
	QString gas();
	/// get stack.
	QString debugStack();
	/// get storage.
	QString debugStorage();
	/// get memory.
	QString debugMemory();
	/// get call data.
	QString debugCallData();
	/// get info to be displayed in the header.
	QString headerInfo();
	/// get end of debug information.
	QString endOfDebug();
	/// get all previous steps.
	QStringList levels();
	/// get the current processed machine state.
	DebuggingState state() { return m_state; }
	/// set the current processed machine state.
	void setState(DebuggingState _state) { m_state = _state;  }
	/// convert all machine state in human readable code.
	static std::tuple<QList<QObject*>, QQMLMap*> getHumanReadableCode(bytes const& _code);

private:
	DebuggingState m_state;
	bytes m_code;
	bytes m_data;
};

}
}
