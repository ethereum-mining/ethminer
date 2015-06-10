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
/** @file main.cpp
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 * EVM Execution tool.
 */
#include <fstream>
#include <iostream>
#include <boost/algorithm/string.hpp>
#include <libdevcore/CommonIO.h>
#include <libdevcore/RLP.h>
#include <libdevcore/SHA3.h>
#include <libethereum/State.h>
#include <libethereum/Executive.h>
#include <libevm/VM.h>
using namespace std;
using namespace dev;
using namespace eth;

void help()
{
	cout
		<< "Usage ethvm <options>" << endl
		<< "Options:" << endl
		;
	exit(0);
}

void version()
{
	cout << "evm version " << dev::Version << endl;
	exit(0);
}

int main(int argc, char** argv)
{
	string incoming = "--";

	State state;
	Address sender = Address(69);
	Address origin = Address(69);
	u256 value = 0;
	u256 gas = state.gasLimitRemaining();
	u256 gasPrice = 0;

	for (int i = 1; i < argc; ++i)
	{
		string arg = argv[i];
		if (arg == "-h" || arg == "--help")
			help();
		else if (arg == "-V" || arg == "--version")
			version();
		else
			incoming = arg;
	}

	bytes code;
	if (incoming == "--" || incoming.empty())
		for (int i = cin.get(); i != -1; i = cin.get())
			code.push_back((char)i);
	else
		code = contents(incoming);
	bytes data = fromHex(boost::trim_copy(asString(code)));
	if (data.empty())
		data = code;

	state.addBalance(sender, value);
	Executive executive(state, eth::LastHashes(), 0);
	ExecutionResult res;
	executive.setResultRecipient(res);
	Transaction t = eth::Transaction(value, gasPrice, gas, data, 0);
	t.forceSender(sender);

	unordered_map<byte, pair<unsigned, bigint>> counts;
	unsigned total = 0;
	bigint memTotal;
	auto onOp = [&](uint64_t, Instruction inst, bigint m, bigint gasCost, bigint, VM*, ExtVMFace const*) {
		counts[(byte)inst].first++;
		counts[(byte)inst].second += gasCost;
		total++;
		if (m > 0)
			memTotal = m;
	};

	executive.initialize(t);
	executive.create(sender, value, gasPrice, gas, &data, origin);
	boost::timer timer;
	executive.go(onOp);
	double execTime = timer.elapsed();
	executive.finalize();
	bytes output = std::move(res.output);
	LogEntries logs = executive.logs();

	cout << "Gas used: " << res.gasUsed << " (+" << t.gasRequired() << " for transaction, -" << res.gasRefunded << " refunded)" << endl;
	cout << "Output: " << toHex(output) << endl;
	cout << logs.size() << " logs" << (logs.empty() ? "." : ":") << endl;
	for (LogEntry const& l: logs)
	{
		cout << "  " << l.address.hex() << ": " << toHex(t.data()) << endl;
		for (h256 const& t: l.topics)
			cout << "    " << t.hex() << endl;
	}

	cout << total << " operations in " << execTime << " seconds." << endl;
	cout << "Maximum memory usage: " << memTotal * 32 << " bytes" << endl;
	cout << "Expensive operations:" << endl;
	for (auto const& c: {Instruction::SSTORE, Instruction::SLOAD, Instruction::CALL, Instruction::CREATE, Instruction::CALLCODE, Instruction::MSTORE8, Instruction::MSTORE, Instruction::MLOAD, Instruction::SHA3})
		if (!!counts[(byte)c].first)
			cout << "  " << instructionInfo(c).name << " x " << counts[(byte)c].first << " (" << counts[(byte)c].second << " gas)" << endl;

	return 0;
}
