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
/** @file Debugger.h
 * @author Gav Wood <i@gavwood.com>
 * @date 2015
 */

#pragma once

#include <libdevcore/RLP.h>
#include <libethcore/Common.h>
#include <libethereum/State.h>
#include <libethereum/Executive.h>
#include <QDialog>
#include <QMap>
#include <QList>
#include "Context.h"

namespace Ui { class Debugger; }

namespace dev
{

namespace az
{

struct WorldState
{
	uint64_t steps;
	dev::Address cur;
	dev::u256 curPC;
	dev::eth::Instruction inst;
	dev::bigint newMemSize;
	dev::u256 gas;
	dev::h256 code;
	dev::h256 callData;
	dev::u256s stack;
	dev::bytes memory;
	dev::bigint gasCost;
	std::unordered_map<dev::u256, dev::u256> storage;
	std::vector<WorldState const*> levels;
};

struct DebugSession
{
	DebugSession() {}

	bool populate(dev::eth::Executive& _executive, dev::eth::Transaction const& _transaction);

	dev::h256 currentCode;
	dev::h256 currentData;
	std::vector<WorldState const*> currentLevels;

	QMap<unsigned, unsigned> pcWarp;
	QList<WorldState> history;

	std::map<dev::u256, dev::bytes> codes;
};

class Debugger: public QDialog
{
	Q_OBJECT

public:
	explicit Debugger(Context* _context, QWidget* _parent = 0);
	~Debugger();

	void populate(dev::eth::Executive& _executive, dev::eth::Transaction const& _transaction);

protected slots:
	void on_callStack_currentItemChanged();
	void on_debugTimeline_valueChanged();
	void on_stepOver_clicked();
	void on_stepInto_clicked();
	void on_stepOut_clicked();
	void on_backOver_clicked();
	void on_backInto_clicked();
	void on_backOut_clicked();
	void on_dump_clicked();
	void on_dumpPretty_clicked();
	void on_dumpStorage_clicked();
	void on_close_clicked() { close(); }

private:
	void init();
	void update();
	void finished();

	void alterDebugStateGroup(bool _enable) const;

	Ui::Debugger* ui;

	DebugSession m_session;
	Context* m_context;
};

}
}
