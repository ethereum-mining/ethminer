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
#include <QPointer>
#include <QQmlEngine>
#include <libevmcore/Instruction.h>
#include <libdevcore/CommonJS.h>
#include <libdevcrypto/Common.h>
#include <libevmcore/Instruction.h>
#include <libdevcore/Common.h>
#include "DebuggingStateWrapper.h"
#include "QBigInt.h"
using namespace dev;
using namespace dev::eth;
using namespace dev::mix;

std::tuple<QList<QObject*>, QQMLMap*> DebuggingStateWrapper::getHumanReadableCode(const bytes& _code)
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
			QPointer<HumanReadableCode> humanCode(new HumanReadableCode(QString::fromStdString(out.str()) + "  "  + s, line));
			codeStr.append(humanCode);
		}
		catch (...)
		{
			qDebug() << QString("Unhandled exception!") << endl <<
					 QString::fromStdString(boost::current_exception_diagnostic_information());
			break;	// probably hit data segment
		}
	}
	return std::make_tuple(codeStr, QPointer<QQMLMap>(new QQMLMap(codeMapping)));
}

QBigInt* DebuggingStateWrapper::gasCost()
{
	return new QBigInt(m_state.gasCost);
}

QBigInt* DebuggingStateWrapper::gas()
{
	return new QBigInt(m_state.gas);
}

QBigInt* DebuggingStateWrapper::newMemSize()
{
	return new QBigInt(m_state.newMemSize);
}

QStringList DebuggingStateWrapper::debugStack()
{
	QStringList stack;
	for (auto i: m_state.stack)
		stack.append(QString::fromStdString(prettyU256(i)));

	return fillList(stack, "");
}

QStringList DebuggingStateWrapper::debugStorage()
{
	QStringList storage;
	for (auto const& i: m_state.storage)
	{
		std::stringstream s;
		s << "@" << prettyU256(i.first) << " " << prettyU256(i.second);
		storage.append(QString::fromStdString(s.str()));
	}
	return fillList(storage, "@ -");
}

QVariantList DebuggingStateWrapper::debugMemory()
{
	std::vector<std::vector<std::string>> dump = memDumpToList(m_state.memory, 16);
	QStringList filled;
	filled.append(" ");
	filled.append(" ");
	filled.append(" ");
	return fillList(qVariantDump(dump), QVariant(filled));
}

QVariantList DebuggingStateWrapper::debugCallData()
{
	std::vector<std::vector<std::string>> dump = memDumpToList(m_data, 16);
	QStringList filled;
	filled.append(" ");
	filled.append(" ");
	filled.append(" ");
	return fillList(qVariantDump(dump), QVariant(filled));
}

std::vector<std::vector<std::string>> DebuggingStateWrapper::memDumpToList(bytes const& _bytes, unsigned _width)
{
	std::vector<std::vector<std::string>> dump;
	for (unsigned i = 0; i < _bytes.size(); i += _width)
	{
		std::stringstream ret;
		std::vector<std::string> dumpLine;
		ret << std::hex << std::setw(4) << std::setfill('0') << i << " ";
		dumpLine.push_back(ret.str());
		ret.str(std::string());
		ret.clear();

		for (unsigned j = i; j < i + _width; ++j)
			if (j < _bytes.size())
				if (_bytes[j] >= 32 && _bytes[j] < 127)
					ret << (char)_bytes[j];
				else
					ret << '?';
			else
				ret << ' ';
		dumpLine.push_back(ret.str());
		ret.str(std::string());
		ret.clear();

		for (unsigned j = i; j < i + _width && j < _bytes.size(); ++j)
			ret << std::setfill('0') << std::setw(2) << std::hex << (unsigned)_bytes[j] << " ";
		dumpLine.push_back(ret.str());
		dump.push_back(dumpLine);
	}
	return dump;
}

QVariantList DebuggingStateWrapper::qVariantDump(std::vector<std::vector<std::string>> const& _dump)
{
	QVariantList ret;
	for (std::vector<std::string> const& line: _dump)
	{
		QStringList qLine;
		for (std::string const& cell: line)
			qLine.push_back(QString::fromStdString(cell));
		ret.append(QVariant(qLine));
	}
	return ret;
}

QStringList DebuggingStateWrapper::fillList(QStringList& _list, QString const& _emptyValue)
{
	if (_list.size() < 20)
	{
		for (int k = _list.size(); k < 20 - _list.size(); k++)
			_list.append(_emptyValue);
	}
	return _list;
}

QVariantList DebuggingStateWrapper::fillList(QVariantList _list, QVariant const& _emptyValue)
{
	if (_list.size() < 20)
	{
		for (int k = _list.size(); k < 20 - _list.size(); k++)
			_list.append(_emptyValue);
	}
	return _list;
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

QString DebuggingStateWrapper::instruction()
{
	return QString::fromStdString(dev::eth::instructionInfo(m_state.inst).name);
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
