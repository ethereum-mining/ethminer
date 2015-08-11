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
/** @file Debugger.cpp
 * @author Gav Wood <i@gavwood.com>
 * @date 2015
 */

#include "Debugger.h"

#include <fstream>
#include <QFileDialog>
#include <libevm/VM.h>
#include <libethereum/ExtVM.h>
#include <libethereum/Executive.h>
#include "ui_Debugger.h"
using namespace std;
using namespace dev;
using namespace az;
using namespace eth;

Debugger::Debugger(Context* _c, QWidget* _parent):
	QDialog(_parent),
	ui(new Ui::Debugger),
	m_context(_c)
{
	ui->setupUi(this);
}

Debugger::~Debugger()
{
	delete ui;
}

void Debugger::init()
{
	if (m_session.history.size())
	{
		alterDebugStateGroup(true);
		ui->debugCode->setEnabled(false);
		ui->debugTimeline->setMinimum(0);
		ui->debugTimeline->setMaximum(m_session.history.size());
		ui->debugTimeline->setValue(0);
	}
}

void Debugger::populate(dev::eth::Executive& _executive, dev::eth::Transaction const& _transaction)
{
	finished();
	if (m_session.populate(_executive, _transaction))
		init();
	update();
}

bool DebugSession::populate(dev::eth::Executive& _executive, dev::eth::Transaction const& _transaction)
{
	try {
		_executive.initialize(_transaction);
		if (_executive.execute())
			return false;
	}
	catch (...)
	{
		// Invalid transaction
		return false;
	}

	vector<WorldState const*> levels;
	bytes lastExtCode;
	bytesConstRef lastData;
	h256 lastHash;
	h256 lastDataHash;
	auto onOp = [&](uint64_t steps, Instruction inst, bigint newMemSize, bigint gasCost, bigint gas, VM* voidVM, ExtVMFace const* voidExt)
	{
		VM& vm = *voidVM;
		ExtVM const& ext = *static_cast<ExtVM const*>(voidExt);
		if (ext.code != lastExtCode)
		{
			lastExtCode = ext.code;
			lastHash = sha3(lastExtCode);
			if (!codes.count(lastHash))
				codes[lastHash] = ext.code;
		}
		if (ext.data != lastData)
		{
			lastData = ext.data;
			lastDataHash = sha3(lastData);
			if (!codes.count(lastDataHash))
				codes[lastDataHash] = ext.data.toBytes();
		}
		if (levels.size() < ext.depth)
			levels.push_back(&history.back());
		else
			levels.resize(ext.depth);
		history.append(WorldState({steps, ext.myAddress, vm.curPC(), inst, newMemSize, static_cast<u256>(gas), lastHash, lastDataHash, vm.stack(), vm.memory(), gasCost, ext.state().storage(ext.myAddress), levels}));
	};
	_executive.go(onOp);
	_executive.finalize();
	return true;
}

void Debugger::finished()
{
	m_session = DebugSession();
	ui->callStack->clear();
	ui->debugCode->clear();
	ui->debugStack->clear();
	ui->debugMemory->setHtml("");
	ui->debugStorage->setHtml("");
	ui->debugStateInfo->setText("");
	alterDebugStateGroup(false);
}

void Debugger::update()
{
	if (m_session.history.size())
	{
		WorldState const& nws = m_session.history[min((int)m_session.history.size() - 1, ui->debugTimeline->value())];
		WorldState const& ws = ui->callStack->currentRow() > 0 ? *nws.levels[nws.levels.size() - ui->callStack->currentRow()] : nws;

		if (ui->debugTimeline->value() >= m_session.history.size())
		{
			if (ws.gasCost > ws.gas)
				ui->debugMemory->setHtml("<h3>OUT-OF-GAS</h3>");
			else if (ws.inst == Instruction::RETURN && ws.stack.size() >= 2)
			{
				unsigned from = (unsigned)ws.stack.back();
				unsigned size = (unsigned)ws.stack[ws.stack.size() - 2];
				unsigned o = 0;
				bytes out(size, 0);
				for (; o < size && from + o < ws.memory.size(); ++o)
					out[o] = ws.memory[from + o];
				ui->debugMemory->setHtml("<h3>RETURN</h3>" + QString::fromStdString(dev::memDump(out, 16, true)));
			}
			else if (ws.inst == Instruction::STOP)
				ui->debugMemory->setHtml("<h3>STOP</h3>");
			else if (ws.inst == Instruction::SUICIDE && ws.stack.size() >= 1)
				ui->debugMemory->setHtml("<h3>SUICIDE</h3>0x" + QString::fromStdString(toString(right160(ws.stack.back()))));
			else
				ui->debugMemory->setHtml("<h3>EXCEPTION</h3>");

			ostringstream ss;
			ss << dec << "EXIT  |  GAS: " << dec << max<dev::bigint>(0, (dev::bigint)ws.gas - ws.gasCost);
			ui->debugStateInfo->setText(QString::fromStdString(ss.str()));
			ui->debugStorage->setHtml("");
			ui->debugCallData->setHtml("");
			m_session.currentData = h256();
			ui->callStack->clear();
			m_session.currentLevels.clear();
			ui->debugCode->clear();
			m_session.currentCode = h256();
			ui->debugStack->setHtml("");
		}
		else
		{
			if (m_session.currentLevels != nws.levels || !ui->callStack->count())
			{
				m_session.currentLevels = nws.levels;
				ui->callStack->clear();
				for (unsigned i = 0; i <= nws.levels.size(); ++i)
				{
					WorldState const& s = i ? *nws.levels[nws.levels.size() - i] : nws;
					ostringstream out;
					out << s.cur.abridged();
					if (i)
						out << " " << instructionInfo(s.inst).name << " @0x" << hex << s.curPC;
					ui->callStack->addItem(QString::fromStdString(out.str()));
				}
			}

			if (ws.code != m_session.currentCode)
			{
				m_session.currentCode = ws.code;
				bytes const& code = m_session.codes[ws.code];
				QListWidget* dc = ui->debugCode;
				dc->clear();
				m_session.pcWarp.clear();
				for (unsigned i = 0; i <= code.size(); ++i)
				{
					byte b = i < code.size() ? code[i] : 0;
					try
					{
						QString s = QString::fromStdString(instructionInfo((Instruction)b).name);
						ostringstream out;
						out << hex << setw(4) << setfill('0') << i;
						m_session.pcWarp[i] = dc->count();
						if (b >= (byte)Instruction::PUSH1 && b <= (byte)Instruction::PUSH32)
						{
							unsigned bc = b - (byte)Instruction::PUSH1 + 1;
							s = "PUSH 0x" + QString::fromStdString(toHex(bytesConstRef(&code[i + 1], bc)));
							i += bc;
						}
						dc->addItem(QString::fromStdString(out.str()) + "  "  + s);
					}
					catch (...)
					{
						cerr << "Unhandled exception!" << endl << boost::current_exception_diagnostic_information();
						break;	// probably hit data segment
					}
				}
			}

			if (ws.callData != m_session.currentData)
			{
				m_session.currentData = ws.callData;
				if (ws.callData)
				{
					assert(m_session.codes.count(ws.callData));
					ui->debugCallData->setHtml(QString::fromStdString(dev::memDump(m_session.codes[ws.callData], 16, true)));
				}
				else
					ui->debugCallData->setHtml("");
			}

			QString stack;
			for (auto i: ws.stack)
				stack.prepend("<div>" + QString::fromStdString(m_context->prettyU256(i)) + "</div>");
			ui->debugStack->setHtml(stack);
			ui->debugMemory->setHtml(QString::fromStdString(dev::memDump(ws.memory, 16, true)));
			assert(m_session.codes.count(ws.code));

			if (m_session.codes[ws.code].size() >= (unsigned)ws.curPC)
			{
				int l = m_session.pcWarp[(unsigned)ws.curPC];
				ui->debugCode->setCurrentRow(max(0, l - 5));
				ui->debugCode->setCurrentRow(min(ui->debugCode->count() - 1, l + 5));
				ui->debugCode->setCurrentRow(l);
			}
			else
				cwarn << "PC (" << (unsigned)ws.curPC << ") is after code range (" << m_session.codes[ws.code].size() << ")";

			ostringstream ss;
			ss << dec << "STEP: " << ws.steps << "  |  PC: 0x" << hex << ws.curPC << "  :  " << instructionInfo(ws.inst).name << "  |  ADDMEM: " << dec << ws.newMemSize << " words  |  COST: " << dec << ws.gasCost <<  "  |  GAS: " << dec << ws.gas;
			ui->debugStateInfo->setText(QString::fromStdString(ss.str()));
			stringstream s;
			for (auto const& i: ws.storage)
				s << "@" << m_context->prettyU256(i.first) << "&nbsp;&nbsp;&nbsp;&nbsp;" << m_context->prettyU256(i.second) << "<br/>";
			ui->debugStorage->setHtml(QString::fromStdString(s.str()));
		}
	}
}

void Debugger::on_callStack_currentItemChanged()
{
	update();
}

void Debugger::alterDebugStateGroup(bool _enable) const
{
	ui->stepOver->setEnabled(_enable);
	ui->stepInto->setEnabled(_enable);
	ui->stepOut->setEnabled(_enable);
	ui->backOver->setEnabled(_enable);
	ui->backInto->setEnabled(_enable);
	ui->backOut->setEnabled(_enable);
	ui->dump->setEnabled(_enable);
	ui->dumpStorage->setEnabled(_enable);
	ui->dumpPretty->setEnabled(_enable);
}

void Debugger::on_debugTimeline_valueChanged()
{
	update();
}

void Debugger::on_stepOver_clicked()
{
	if (ui->debugTimeline->value() < m_session.history.size()) {
		auto l = m_session.history[ui->debugTimeline->value()].levels.size();
		if ((ui->debugTimeline->value() + 1) < m_session.history.size() && m_session.history[ui->debugTimeline->value() + 1].levels.size() > l)
		{
			on_stepInto_clicked();
			if (m_session.history[ui->debugTimeline->value()].levels.size() > l)
				on_stepOut_clicked();
		}
		else
			on_stepInto_clicked();
	}
}

void Debugger::on_stepInto_clicked()
{
	ui->debugTimeline->setValue(ui->debugTimeline->value() + 1);
	ui->callStack->setCurrentRow(0);
}

void Debugger::on_stepOut_clicked()
{
	if (ui->debugTimeline->value() < m_session.history.size())
	{
		auto ls = m_session.history[ui->debugTimeline->value()].levels.size();
		auto l = ui->debugTimeline->value();
		for (; l < m_session.history.size() && m_session.history[l].levels.size() >= ls; ++l) {}
		ui->debugTimeline->setValue(l);
		ui->callStack->setCurrentRow(0);
	}
}

void Debugger::on_backInto_clicked()
{
	ui->debugTimeline->setValue(ui->debugTimeline->value() - 1);
	ui->callStack->setCurrentRow(0);
}

void Debugger::on_backOver_clicked()
{
	auto l = m_session.history[ui->debugTimeline->value()].levels.size();
	if (ui->debugTimeline->value() > 0 && m_session.history[ui->debugTimeline->value() - 1].levels.size() > l)
	{
		on_backInto_clicked();
		if (m_session.history[ui->debugTimeline->value()].levels.size() > l)
			on_backOut_clicked();
	}
	else
		on_backInto_clicked();
}

void Debugger::on_backOut_clicked()
{
	if (ui->debugTimeline->value() > 0 && m_session.history.size() > 0)
	{
		auto ls = m_session.history[min(ui->debugTimeline->value(), m_session.history.size() - 1)].levels.size();
		int l = ui->debugTimeline->value();
		for (; l > 0 && m_session.history[l].levels.size() >= ls; --l) {}
		ui->debugTimeline->setValue(l);
		ui->callStack->setCurrentRow(0);
	}
}

void Debugger::on_dump_clicked()
{
	QString fn = QFileDialog::getSaveFileName(this, "Select file to output EVM trace");
	ofstream f(fn.toStdString());
	if (f.is_open())
		for (WorldState const& ws: m_session.history)
			f << ws.cur << " " << hex << toHex(dev::toCompactBigEndian(ws.curPC, 1)) << " " << hex << toHex(dev::toCompactBigEndian((int)(byte)ws.inst, 1)) << " " << hex << toHex(dev::toCompactBigEndian((uint64_t)ws.gas, 1)) << endl;
}

void Debugger::on_dumpPretty_clicked()
{
	QString fn = QFileDialog::getSaveFileName(this, "Select file to output EVM trace");
	ofstream f(fn.toStdString());
	if (f.is_open())
		for (WorldState const& ws: m_session.history)
		{
			f << endl << "    STACK" << endl;
			for (auto i: ws.stack)
				f << (h256)i << endl;
			f << "    MEMORY" << endl << dev::memDump(ws.memory);
			f << "    STORAGE" << endl;
			for (auto const& i: ws.storage)
				f << showbase << hex << i.first << ": " << i.second << endl;
			f << dec << ws.levels.size() << " | " << ws.cur << " | #" << ws.steps << " | " << hex << setw(4) << setfill('0') << ws.curPC << " : " << instructionInfo(ws.inst).name << " | " << dec << ws.gas << " | -" << dec << ws.gasCost << " | " << ws.newMemSize << "x32";
		}
}

void Debugger::on_dumpStorage_clicked()
{
	QString fn = QFileDialog::getSaveFileName(this, "Select file to output EVM trace");
	ofstream f(fn.toStdString());
	if (f.is_open())
		for (WorldState const& ws: m_session.history)
		{
			if (ws.inst == Instruction::STOP || ws.inst == Instruction::RETURN || ws.inst == Instruction::SUICIDE)
				for (auto i: ws.storage)
					f << toHex(dev::toCompactBigEndian(i.first, 1)) << " " << toHex(dev::toCompactBigEndian(i.second, 1)) << endl;
			f << ws.cur << " " << hex << toHex(dev::toCompactBigEndian(ws.curPC, 1)) << " " << hex << toHex(dev::toCompactBigEndian((int)(byte)ws.inst, 1)) << " " << hex << toHex(dev::toCompactBigEndian((uint64_t)ws.gas, 1)) << endl;
		}
}
