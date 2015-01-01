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

#include <QApplication>
#include <QDebug>
#include "libevmcore/Instruction.h"
#include "libdevcore/CommonJS.h"
#include "libdevcrypto/Common.h"
#include "libevmcore/Instruction.h"
#include "libdevcore/Common.h"
#include "DebuggingStateWrapper.h"
using namespace dev;
using namespace dev::eth;
using namespace dev::mix;

std::tuple<QList<QObject*>, QQMLMap*> DebuggingStateWrapper::getHumanReadableCode(const bytes& _code, QObject* _objUsedAsParent)
{
	QList<QObject*> codeStr;
	QMap<int, int> codeMapping;
	for (unsigned i = 0; i <= _code.size(); ++i)
	{
		byte b = i < _code.size() ? _code[i] : 0;
		try
		{
			QString s = QString::fromStdString(instructionInfo((Instruction)b).name);
			std::ostringstream out;
			out << std::hex << std::setw(4) << std::setfill('0') << i;
			codeMapping[i] = codeStr.size();
			int line = i;
			if (b >= (byte)Instruction::PUSH1 && b <= (byte)Instruction::PUSH32)
			{
				unsigned bc = getPushNumber((Instruction)b);
				s = "PUSH 0x" + QString::fromStdString(toHex(bytesConstRef(&_code[i + 1], bc)));
				i += bc;
			}
			HumanReadableCode* humanCode = new HumanReadableCode(QString::fromStdString(out.str()) + "  "  + s, line, _objUsedAsParent);
			codeStr.append(humanCode);
		}
		catch (...)
		{
			qDebug() << QString("Unhandled exception!") << endl <<
					 QString::fromStdString(boost::current_exception_diagnostic_information());
			break;	// probably hit data segment
		}
	}
	return std::make_tuple(codeStr, new QQMLMap(codeMapping, _objUsedAsParent));
}

QString DebuggingStateWrapper::gasLeft()
{
	std::ostringstream ss;
	ss << std::dec << (m_state.gas - m_state.gasCost);
	return QString::fromStdString(ss.str());
}

QString DebuggingStateWrapper::gasCost()
{
	std::ostringstream ss;
	ss << std::dec << m_state.gasCost;
	return QString::fromStdString(ss.str());
}

QString DebuggingStateWrapper::gas()
{
	std::ostringstream ss;
	ss << std::dec << m_state.gas;
	return QString::fromStdString(ss.str());
}

QString DebuggingStateWrapper::debugStack()
{
	QString stack;
	for (auto i: m_state.stack)
		stack.prepend(QString::fromStdString(prettyU256(i)) + "\n");

	return stack;
}

QString DebuggingStateWrapper::debugStorage()
{
	std::stringstream s;
	for (auto const& i: m_state.storage)
		s << "@" << prettyU256(i.first) << "&nbsp;&nbsp;&nbsp;&nbsp;" << prettyU256(i.second);

	return QString::fromStdString(s.str());
}

QString DebuggingStateWrapper::debugMemory()
{
	return QString::fromStdString(memDump(m_state.memory, 16, false));
}

QString DebuggingStateWrapper::debugCallData()
{
	return QString::fromStdString(memDump(m_data, 16, false));
}

QStringList DebuggingStateWrapper::levels()
{
	QStringList levelsStr;
	for (unsigned i = 0; i <= m_state.levels.size(); ++i)
	{
		std::ostringstream out;
		out << m_state.cur.abridged();
		if (i)
			out << " " << instructionInfo(m_state.inst).name << " @0x" << std::hex << m_state.curPC;
		levelsStr.append(QString::fromStdString(out.str()));
	}
	return levelsStr;
}

QString DebuggingStateWrapper::headerInfo()
{
	std::ostringstream ss;
	ss << std::dec << " " << QApplication::tr("STEP").toStdString() << " : " << m_state.steps << "  |  PC: 0x" << std::hex << m_state.curPC << "  :  " << dev::eth::instructionInfo(m_state.inst).name << "  |  ADDMEM: " << std::dec << m_state.newMemSize << " " << QApplication::tr("words").toStdString() << " | " << QApplication::tr("COST").toStdString() << " : " << std::dec << m_state.gasCost <<  "  | " << QApplication::tr("GAS").toStdString() << " : " << std::dec << m_state.gas;
	return QString::fromStdString(ss.str());
}

QString DebuggingStateWrapper::endOfDebug()
{
	if (m_state.gasCost > m_state.gas)
		return QApplication::tr("OUT-OF-GAS");
	else if (m_state.inst == Instruction::RETURN && m_state.stack.size() >= 2)
	{
		unsigned from = (unsigned)m_state.stack.back();
		unsigned size = (unsigned)m_state.stack[m_state.stack.size() - 2];
		unsigned o = 0;
		bytes out(size, 0);
		for (; o < size && from + o < m_state.memory.size(); ++o)
			out[o] = m_state.memory[from + o];
		return QApplication::tr("RETURN") + " " + QString::fromStdString(dev::memDump(out, 16, false));
	}
	else if (m_state.inst == Instruction::STOP)
		return QApplication::tr("STOP");
	else if (m_state.inst == Instruction::SUICIDE && m_state.stack.size() >= 1)
		return QApplication::tr("SUICIDE") + " 0x" + QString::fromStdString(toString(right160(m_state.stack.back())));
	else
		return QApplication::tr("EXCEPTION");
}
