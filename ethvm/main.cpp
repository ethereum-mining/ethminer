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
#include <ctime>
#include <boost/algorithm/string.hpp>
#include <libdevcore/CommonIO.h>
#include <libdevcore/RLP.h>
#include <libdevcore/SHA3.h>
#include <libethereum/Block.h>
#include <libethereum/Executive.h>
#include <libevm/VM.h>
#include <libevm/VMFactory.h>
using namespace std;
using namespace dev;
using namespace eth;

void help()
{
	cout
		<< "Usage ethvm <options> [trace|stats|output] (<file>|--)" << endl
		<< "Transaction options:" << endl
		<< "    --value <n>  Transaction should transfer the <n> wei (default: 0)." << endl
		<< "    --gas <n>  Transaction should be given <n> gas (default: block gas limit)." << endl
		<< "    --gas-price <n>  Transaction's gas price' should be <n> (default: 0)." << endl
		<< "    --sender <a>  Transaction sender should be <a> (default: 0000...0069)." << endl
		<< "    --origin <a>  Transaction origin should be <a> (default: 0000...0069)." << endl
#if ETH_EVMJIT || !ETH_TRUE
		<< endl
		<< "VM options:" << endl
		<< "    --vm <vm-kind>  Select VM. Options are: interpreter, jit, smart. (default: interpreter)" << endl
#endif
		<< endl
		<< "Options for trace:" << endl
		<< "    --flat  Minimal whitespace in the JSON." << endl
		<< "    --mnemonics  Show instruction mnemonics in the trace (non-standard)." << endl
		<< endl
		<< "General options:" << endl
		<< "    -V,--version  Show the version and exit." << endl
		<< "    -h,--help  Show this help message and exit." << endl;
	exit(0);
}

void version()
{
	cout << "ethvm version " << dev::Version << endl;
	cout << "By Gav Wood, 2015." << endl;
	cout << "Build: " << DEV_QUOTED(ETH_BUILD_PLATFORM) << "/" << DEV_QUOTED(ETH_BUILD_TYPE) << endl;
	exit(0);
}

enum class Mode
{
	Trace,
	Statistics,
	OutputOnly
};

int main(int argc, char** argv)
{
	string incoming = "--";

	Mode mode = Mode::Statistics;
	State state;
	Address sender = Address(69);
	Address origin = Address(69);
	u256 value = 0;
	u256 gas = Block().gasLimitRemaining();
	u256 gasPrice = 0;
	bool styledJson = true;
	StandardTrace st;
	EnvInfo envInfo;

	for (int i = 1; i < argc; ++i)
	{
		string arg = argv[i];
		if (arg == "-h" || arg == "--help")
			help();
		else if (arg == "-V" || arg == "--version")
			version();
#if ETH_EVMJIT
		else if (arg == "--vm" && i + 1 < argc)
		{
			string vmKind = argv[++i];
			if (vmKind == "interpreter")
				VMFactory::setKind(VMKind::Interpreter);
			else if (vmKind == "jit")
				VMFactory::setKind(VMKind::JIT);
			else if (vmKind == "smart")
				VMFactory::setKind(VMKind::Smart);
			else
			{
				cerr << "Unknown VM kind: " << vmKind << endl;
				return -1;
			}
		}
#endif
		else if (arg == "--mnemonics")
			st.setShowMnemonics();
		else if (arg == "--flat")
			styledJson = false;
		else if (arg == "--value" && i + 1 < argc)
			value = u256(argv[++i]);
		else if (arg == "--sender" && i + 1 < argc)
			sender = Address(argv[++i]);
		else if (arg == "--origin" && i + 1 < argc)
			origin = Address(argv[++i]);
		else if (arg == "--gas" && i + 1 < argc)
			gas = u256(argv[++i]);
		else if (arg == "--gas-price" && i + 1 < argc)
			gasPrice = u256(argv[++i]);
		else if (arg == "--value" && i + 1 < argc)
			value = u256(argv[++i]);
		else if (arg == "--value" && i + 1 < argc)
			value = u256(argv[++i]);
		else if (arg == "--beneficiary" && i + 1 < argc)
			envInfo.setBeneficiary(Address(argv[++i]));
		else if (arg == "--number" && i + 1 < argc)
			envInfo.setNumber(u256(argv[++i]));
		else if (arg == "--difficulty" && i + 1 < argc)
			envInfo.setDifficulty(u256(argv[++i]));
		else if (arg == "--timestamp" && i + 1 < argc)
			envInfo.setTimestamp(u256(argv[++i]));
		else if (arg == "--gas-limit" && i + 1 < argc)
			envInfo.setGasLimit(u256(argv[++i]));
		else if (arg == "--value" && i + 1 < argc)
			value = u256(argv[++i]);
		else if (arg == "stats")
			mode = Mode::Statistics;
		else if (arg == "output")
			mode = Mode::OutputOnly;
		else if (arg == "trace")
			mode = Mode::Trace;
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
	Executive executive(state, envInfo);
	ExecutionResult res;
	executive.setResultRecipient(res);
	Transaction t = eth::Transaction(value, gasPrice, gas, data, 0);
	t.forceSender(sender);

	unordered_map<byte, pair<unsigned, bigint>> counts;
	unsigned total = 0;
	bigint memTotal;
	auto onOp = [&](uint64_t step, Instruction inst, bigint m, bigint gasCost, bigint gas, VM* vm, ExtVMFace const* extVM) {
		if (mode == Mode::Statistics)
		{
			counts[(byte)inst].first++;
			counts[(byte)inst].second += gasCost;
			total++;
			if (m > 0)
				memTotal = m;
		}
		else if (mode == Mode::Trace)
			st(step, inst, m, gasCost, gas, vm, extVM);
	};

	executive.initialize(t);
	executive.create(sender, value, gasPrice, gas, &data, origin);
	Timer timer;
	executive.go(onOp);
	double execTime = timer.elapsed();
	executive.finalize();
	bytes output = std::move(res.output);

	if (mode == Mode::Statistics)
	{
		cout << "Gas used: " << res.gasUsed << " (+" << t.gasRequired() << " for transaction, -" << res.gasRefunded << " refunded)" << endl;
		cout << "Output: " << toHex(output) << endl;
		LogEntries logs = executive.logs();
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
	}
	else if (mode == Mode::Trace)
		cout << st.json(styledJson);
	else if (mode == Mode::OutputOnly)
		cout << toHex(output);

	return 0;
}
