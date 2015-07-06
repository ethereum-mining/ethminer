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

// Make sure boost/asio.hpp is included before windows.h.
#include <boost/asio.hpp>

#include <QStringList>
#include <QHash>
#include <libdevcore/Common.h>
#include <libethereum/State.h>
#include <libethereum/Executive.h>
#include "MachineStates.h"
#include "QVariableDefinition.h"
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

public:
	QInstruction(QObject* _owner, QString _line): QObject(_owner), m_line(_line) {}

private:
	QString m_line;
};

/**
 * @brief Solidity state
 */
class QSolState: public QObject
{
	Q_OBJECT
	Q_PROPERTY(QVariantMap storage MEMBER m_storage CONSTANT)
	Q_PROPERTY(QVariantList callStack MEMBER m_callStack CONSTANT)
	Q_PROPERTY(QVariantMap locals MEMBER m_locals CONSTANT)
	Q_PROPERTY(int start MEMBER m_start CONSTANT)
	Q_PROPERTY(int end MEMBER m_end CONSTANT)
	Q_PROPERTY(QString sourceName MEMBER m_sourceName CONSTANT)

public:
	QSolState(QObject* _parent, QVariantMap&& _storage, QVariantList&& _callStack, QVariantMap&& _locals, int _start, int _end, QString _sourceName):
		QObject(_parent), m_storage(std::move(_storage)), m_callStack(std::move(_callStack)), m_locals(std::move(_locals)), m_start(_start), m_end(_end), m_sourceName(_sourceName)
	{ }

private:
	QVariantMap m_storage;
	QVariantList m_callStack;
	QVariantMap m_locals;
	int m_start;
	int m_end;
	QString m_sourceName;
};

/**
 * @brief Shared container for lines
 */
class QCode: public QObject
{
	Q_OBJECT
	Q_PROPERTY(QVariantList instructions MEMBER m_instructions CONSTANT)
	Q_PROPERTY(QString address MEMBER m_address CONSTANT)
	Q_PROPERTY(QString documentId MEMBER m_document CONSTANT)

public:
	QCode(QObject* _owner, QString const& _address, QVariantList&& _instrunctions): QObject(_owner), m_instructions(std::move(_instrunctions)), m_address(_address) {}
	void setDocument(QString const& _documentId) { m_document = _documentId; }

private:
	QVariantList m_instructions;
	QString m_address;
	QString m_document;
};

/**
 * @brief Shared container for call data
 */
class QCallData: public QObject
{
	Q_OBJECT
	Q_PROPERTY(QVariantList items MEMBER m_items CONSTANT)

public:
	QCallData(QObject* _owner, QVariantList&& _items): QObject(_owner), m_items(std::move(_items)) {}

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
	void setStates(QVariantList&& _states) { m_states = std::move(_states); }

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
	Q_PROPERTY(int instructionIndex MEMBER m_instructionIndex CONSTANT)
	Q_PROPERTY(QBigInt* gasCost READ gasCost CONSTANT)
	Q_PROPERTY(QBigInt* gas READ gas CONSTANT)
	Q_PROPERTY(QString instruction READ instruction CONSTANT)
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
	Q_PROPERTY(QObject* solidity MEMBER m_solState CONSTANT)

public:
	QMachineState(QObject* _owner, int _instructionIndex, MachineState const& _state, QCode* _code, QCallData* _callData, QSolState* _solState):
		QObject(_owner), m_instructionIndex(_instructionIndex), m_state(_state), m_code(_code), m_callData(_callData), m_solState(_solState) { }
	/// Get the step of this machine states.
	int step() { return  (int)m_state.steps; }
	/// Get the proccessed code index.
	int curPC() { return (int)m_state.curPC; }
	/// Get the code id
	unsigned codeIndex() { return m_state.codeIndex; }
	/// Get the call data id
	unsigned dataIndex() { return m_state.dataIndex; }
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
	/// Get end of debug information.
	QString endOfDebug();
	/// Get the new memory size.
	QBigInt* newMemSize();
	/// Get current instruction
	QString instruction();
	/// Get all previous steps.
	QVariantList levels();
	/// Get the current processed machine state.
	MachineState const& state() const { return m_state; }
	/// Convert all machine states in human readable code.
	static QCode* getHumanReadableCode(QObject* _owner, const Address& _address, const bytes& _code, QHash<int, int>& o_codeMap);
	/// Convert call data into human readable form
	static QCallData* getDebugCallData(QObject* _owner, bytes const& _data);

private:
	int m_instructionIndex;
	MachineState m_state;
	QCode* m_code;
	QCallData* m_callData;
	QSolState* m_solState;
};

}
}
