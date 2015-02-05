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
/** @file DebuggingStateWrapper.cpp
 * @author Yann yann@ethdev.com
 * @date 2014
 * Used to translate c++ type (u256, bytes, ...) into friendly value (to be used by QML).
 */

#include <tuple>
#include <QDebug>
#include <QPointer>
#include <QQmlEngine>
#include <QVariantList>
#include <libevmcore/Instruction.h>
#include <libethcore/CommonJS.h>
#include <libdevcrypto/Common.h>
#include <libevmcore/Instruction.h>
#include <libdevcore/Common.h>
#include "DebuggingStateWrapper.h"
#include "QBigInt.h"
using namespace dev;
using namespace dev::eth;
using namespace dev::mix;

namespace
{
	static QVariantList memDumpToList(bytes const& _bytes, unsigned _width)
	{
		QVariantList dumpList;
		for (unsigned i = 0; i < _bytes.size(); i += _width)
		{
			std::stringstream ret;

			for (unsigned j = i; j < i + _width; ++j)
				if (j < _bytes.size())
					if (_bytes[j] >= 32 && _bytes[j] < 127)
						ret << (char)_bytes[j];
					else
						ret << '?';
				else
					ret << ' ';
			QString strPart = QString::fromStdString(ret.str());

			ret.clear();
			ret.str(std::string());

			for (unsigned j = i; j < i + _width && j < _bytes.size(); ++j)
				ret << std::setfill('0') << std::setw(2) << std::hex << (unsigned)_bytes[j] << " ";
			QString hexPart = QString::fromStdString(ret.str());

			QStringList line = { strPart, hexPart };
			dumpList.push_back(line);
		}
		return dumpList;
	}
}

QCode* QMachineState::getHumanReadableCode(QObject* _owner, const bytes& _code)
{
	QVariantList codeStr;
	for (unsigned i = 0; i <= _code.size(); ++i)
	{
		byte b = i < _code.size() ? _code[i] : 0;
		try
		{
			QString s = QString::fromStdString(instructionInfo((Instruction)b).name);
			std::ostringstream out;
			out << std::hex << std::setw(4) << std::setfill('0') << i;
			int line = i;
			if (b >= (byte)Instruction::PUSH1 && b <= (byte)Instruction::PUSH32)
			{
				unsigned bc = getPushNumber((Instruction)b);
				s = "PUSH 0x" + QString::fromStdString(toHex(bytesConstRef(&_code[i + 1], bc)));
				i += bc;
			}
			codeStr.append(QVariant::fromValue(new QInstruction(_owner, QString::fromStdString(out.str()) + "  "  + s, line)));
		}
		catch (...)
		{
			qDebug() << QString("Unhandled exception!") << endl <<
					 QString::fromStdString(boost::current_exception_diagnostic_information());
			break;	// probably hit data segment
		}
	}
	return new QCode(_owner, std::move(codeStr));
}

QBigInt* QMachineState::gasCost()
{
	return new QBigInt(m_state.gasCost);
}

QBigInt* QMachineState::gas()
{
	return new QBigInt(m_state.gas);
}

QBigInt* QMachineState::newMemSize()
{
	return new QBigInt(m_state.newMemSize);
}

QStringList QMachineState::debugStack()
{
	QStringList stack;
	for (std::vector<u256>::reverse_iterator i = m_state.stack.rbegin(); i != m_state.stack.rend(); ++i)
		stack.append(QString::fromStdString(prettyU256(*i)));
	return stack;
}

QStringList QMachineState::debugStorage()
{
	QStringList storage;
	for (auto const& i: m_state.storage)
	{
		std::stringstream s;
		s << "@" << prettyU256(i.first) << "\t" << prettyU256(i.second);
		storage.append(QString::fromStdString(s.str()));
	}
	return storage;
}

QVariantList QMachineState::debugMemory()
{
	return memDumpToList(m_state.memory, 16);
}

QCallData* QMachineState::getDebugCallData(QObject* _owner, bytes const& _data)
{
	return new QCallData(_owner, memDumpToList(_data, 16));
}

QVariantList QMachineState::levels()
{
	QVariantList levelList;
	for (unsigned l: m_state.levels)
		levelList.push_back(l);
	return levelList;
}

QString QMachineState::address()
{
	return QString::fromStdString(toString(m_state.address));
}

QString QMachineState::instruction()
{
	return QString::fromStdString(dev::eth::instructionInfo(m_state.inst).name);
}

QString QMachineState::endOfDebug()
{
	if (m_state.gasCost > m_state.gas)
		return QObject::tr("OUT-OF-GAS");
	else if (m_state.inst == Instruction::RETURN && m_state.stack.size() >= 2)
	{
		unsigned from = (unsigned)m_state.stack.back();
		unsigned size = (unsigned)m_state.stack[m_state.stack.size() - 2];
		unsigned o = 0;
		bytes out(size, 0);
		for (; o < size && from + o < m_state.memory.size(); ++o)
			out[o] = m_state.memory[from + o];
		return QObject::tr("RETURN") + " " + QString::fromStdString(dev::memDump(out, 16, false));
	}
	else if (m_state.inst == Instruction::STOP)
		return QObject::tr("STOP");
	else if (m_state.inst == Instruction::SUICIDE && m_state.stack.size() >= 1)
		return QObject::tr("SUICIDE") + " 0x" + QString::fromStdString(toString(right160(m_state.stack.back())));
	else
		return QObject::tr("EXCEPTION");
}
