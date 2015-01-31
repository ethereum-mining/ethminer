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
class QInstruction: public QObject
{
	Q_OBJECT
	Q_PROPERTY(QString line MEMBER m_line CONSTANT)
	Q_PROPERTY(int processIndex MEMBER m_processIndex CONSTANT)

public:
	QInstruction(QObject* _owner, QString _line, int _processIndex): QObject(_owner), m_line(_line), m_processIndex(_processIndex) {}

private:
	QString m_line;
	int m_processIndex;
};

/**
 * @brief Shared container for lines
 */
class QCode: public QObject
{
	Q_OBJECT
	Q_PROPERTY(QVariantList instructions MEMBER m_instructions CONSTANT)

public:
	QCode(QObject* _owner, QVariantList&& _instrunctions): QObject(_owner), m_instructions(_instrunctions) {}

private:
	QVariantList m_instructions;
};

/**
 * @brief Shared container for call data
 */
class QCallData: public QObject
{
	Q_OBJECT
	Q_PROPERTY(QVariantList items MEMBER m_items CONSTANT)

public:
	QCallData(QObject* _owner, QVariantList&& _items): QObject(_owner), m_items(_items) {}

private:
	QVariantList m_items;
};

/**
 * @brief Shared container for machine states
 */
class QDebugData: public QObject
{
	Q_OBJECT
	Q_PROPERTY(QVariantList states MEMBER m_states CONSTANT)

public:
	QDebugData() { }
	void setStates(QVariantList&& _states) { m_states = _states; }

private:
	QVariantList m_states;
};

/**
 * @brief Wrap MachineState in QObject
 */
class QMachineState: public QObject
{
	Q_OBJECT
	Q_PROPERTY(int step READ step CONSTANT)
	Q_PROPERTY(int curPC READ curPC CONSTANT)
	Q_PROPERTY(QBigInt* gasCost READ gasCost CONSTANT)
	Q_PROPERTY(QBigInt* gas READ gas CONSTANT)
	Q_PROPERTY(QString instruction READ instruction CONSTANT)
	Q_PROPERTY(QString address READ address CONSTANT)
	Q_PROPERTY(QStringList debugStack READ debugStack CONSTANT)
	Q_PROPERTY(QStringList debugStorage READ debugStorage CONSTANT)
	Q_PROPERTY(QVariantList debugMemory READ debugMemory CONSTANT)
	Q_PROPERTY(QObject* code MEMBER m_code CONSTANT)
	Q_PROPERTY(QObject* callData MEMBER m_callData CONSTANT)
	Q_PROPERTY(QString endOfDebug READ endOfDebug CONSTANT)
	Q_PROPERTY(QBigInt* newMemSize READ newMemSize CONSTANT)
	Q_PROPERTY(QVariantList levels READ levels CONSTANT)
	Q_PROPERTY(unsigned codeIndex READ codeIndex CONSTANT)
	Q_PROPERTY(unsigned dataIndex READ dataIndex CONSTANT)

public:
	QMachineState(QObject* _owner, MachineState const& _state, QCode* _code, QCallData* _callData):
		QObject(_owner), m_state(_state), m_code(_code), m_callData(_callData) {}
	/// Get the step of this machine states.
	int step() { return  (int)m_state.steps; }
	/// Get the proccessed code index.
	int curPC() { return (int)m_state.curPC; }
	/// Get the code id
	unsigned codeIndex() { return m_state.codeIndex; }
	/// Get the call data id
	unsigned dataIndex() { return m_state.dataIndex; }
	/// Get address for call stack
	QString address();
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
	/// get end of debug information.
	QString endOfDebug();
	/// Get the new memory size.
	QBigInt* newMemSize();
	/// Get current instruction
	QString instruction();
	/// Get all previous steps.
	QVariantList levels();
	/// Get the current processed machine state.
	MachineState state() { return m_state; }
	/// Set the current processed machine state.
	void setState(MachineState _state) { m_state = _state;  }
	/// Convert all machine states in human readable code.
	static QCode* getHumanReadableCode(QObject* _owner, bytes const& _code);
	/// Convert call data into human readable form
	static QCallData* getDebugCallData(QObject* _owner, bytes const& _data);

private:
	MachineState m_state;
	QCode* m_code;
	QCallData* m_callData;
};

}
}
