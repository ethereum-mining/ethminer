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
#include "libethereum/State.h"
#include "libethereum/Executive.h"
#include "libdevcore/Common.h"

namespace dev
{
namespace mix
{

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

struct DebuggingContent
{
	QList<DebuggingState> states;
	bytes executionCode;
	bytesConstRef executionData;
	bool contentAvailable;
	QString message;
};

/**
 * @brief Contains the line nb of the assembly code and the corresponding index in the code bytes array.
 */
class HumanReadableCode: public QObject
{
	Q_OBJECT
	Q_PROPERTY(QString line READ line)
	Q_PROPERTY(int processIndex READ processIndex)

public:
	HumanReadableCode(QString _line, int _processIndex, QObject* _parent): QObject(_parent), m_line(_line), m_processIndex(_processIndex) {}
	QString line() { return m_line; }
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
	QQMLMap(QMap<int, int> _map, QObject* _parent): QObject(_parent), m_map(_map) { }
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
	Q_PROPERTY(int step READ step)
	Q_PROPERTY(int curPC READ curPC)
	Q_PROPERTY(QString gasCost READ gasCost)
	Q_PROPERTY(QString gas READ gas)
	Q_PROPERTY(QString gasLeft READ gasLeft)
	Q_PROPERTY(QString debugStack READ debugStack)
	Q_PROPERTY(QString debugStorage READ debugStorage)
	Q_PROPERTY(QString debugMemory READ debugMemory)
	Q_PROPERTY(QString debugCallData READ debugCallData)
	Q_PROPERTY(QString headerInfo READ headerInfo)
	Q_PROPERTY(QString endOfDebug READ endOfDebug)
	Q_PROPERTY(QStringList levels READ levels)

public:
	DebuggingStateWrapper(bytes _code, bytes _data, QObject* _parent): QObject(_parent), m_code(_code), m_data(_data) {}
	int step() { return  (int)m_state.steps; }
	int curPC() { return (int)m_state.curPC; }
	QString gasLeft();
	QString gasCost();
	QString gas();
	QString debugStack();
	QString debugStorage();
	QString debugMemory();
	QString debugCallData();
	QString headerInfo();
	QString endOfDebug();
	QStringList levels();
	DebuggingState state() { return m_state; }
	void setState(DebuggingState _state) { m_state = _state;  }
	static std::tuple<QList<QObject*>, QQMLMap*> getHumanReadableCode(bytes const& _code, QObject* _objUsedAsParent);

private:
	DebuggingState m_state;
	bytes m_code;
	bytes m_data;
};

}

}
