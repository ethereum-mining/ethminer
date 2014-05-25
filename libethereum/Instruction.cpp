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
/** @file Instruction.cpp
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 */

#include "Instruction.h"

#include <boost/algorithm/string.hpp>
#include <boost/spirit/include/qi.hpp>
#include <boost/spirit/include/phoenix.hpp>
#include <boost/spirit/include/support_utree.hpp>
#include <libethcore/Log.h>
#include "CommonEth.h"
using namespace std;
using namespace eth;
namespace qi = boost::spirit::qi;
namespace px = boost::phoenix;
namespace sp = boost::spirit;

const std::map<std::string, Instruction> eth::c_instructions =
{
	{ "STOP", Instruction::STOP },
	{ "ADD", Instruction::ADD },
	{ "SUB", Instruction::SUB },
	{ "MUL", Instruction::MUL },
	{ "DIV", Instruction::DIV },
	{ "SDIV", Instruction::SDIV },
	{ "MOD", Instruction::MOD },
	{ "SMOD", Instruction::SMOD },
	{ "EXP", Instruction::EXP },
	{ "NEG", Instruction::NEG },
	{ "LT", Instruction::LT },
	{ "GT", Instruction::GT },
	{ "SLT", Instruction::SLT },
	{ "SGT", Instruction::SGT },
	{ "EQ", Instruction::EQ },
	{ "NOT", Instruction::NOT },
	{ "AND", Instruction::AND },
	{ "OR", Instruction::OR },
	{ "XOR", Instruction::XOR },
	{ "BYTE", Instruction::BYTE },
	{ "SHA3", Instruction::SHA3 },
	{ "ADDRESS", Instruction::ADDRESS },
	{ "BALANCE", Instruction::BALANCE },
	{ "ORIGIN", Instruction::ORIGIN },
	{ "CALLER", Instruction::CALLER },
	{ "CALLVALUE", Instruction::CALLVALUE },
	{ "CALLDATALOAD", Instruction::CALLDATALOAD },
	{ "CALLDATASIZE", Instruction::CALLDATASIZE },
	{ "CALLDATACOPY", Instruction::CALLDATACOPY },
	{ "CODESIZE", Instruction::CODESIZE },
	{ "CODECOPY", Instruction::CODECOPY },
	{ "GASPRICE", Instruction::GASPRICE },
	{ "PREVHASH", Instruction::PREVHASH },
	{ "COINBASE", Instruction::COINBASE },
	{ "TIMESTAMP", Instruction::TIMESTAMP },
	{ "NUMBER", Instruction::NUMBER },
	{ "DIFFICULTY", Instruction::DIFFICULTY },
	{ "GASLIMIT", Instruction::GASLIMIT },
	{ "POP", Instruction::POP },
	{ "DUP", Instruction::DUP },
	{ "SWAP", Instruction::SWAP },
	{ "MLOAD", Instruction::MLOAD },
	{ "MSTORE", Instruction::MSTORE },
	{ "MSTORE8", Instruction::MSTORE8 },
	{ "SLOAD", Instruction::SLOAD },
	{ "SSTORE", Instruction::SSTORE },
	{ "JUMP", Instruction::JUMP },
	{ "JUMPI", Instruction::JUMPI },
	{ "PC", Instruction::PC },
	{ "MEMSIZE", Instruction::MEMSIZE },
	{ "GAS", Instruction::GAS },
	{ "PUSH1", Instruction::PUSH1 },
	{ "PUSH2", Instruction::PUSH2 },
	{ "PUSH3", Instruction::PUSH3 },
	{ "PUSH4", Instruction::PUSH4 },
	{ "PUSH5", Instruction::PUSH5 },
	{ "PUSH6", Instruction::PUSH6 },
	{ "PUSH7", Instruction::PUSH7 },
	{ "PUSH8", Instruction::PUSH8 },
	{ "PUSH9", Instruction::PUSH9 },
	{ "PUSH10", Instruction::PUSH10 },
	{ "PUSH11", Instruction::PUSH11 },
	{ "PUSH12", Instruction::PUSH12 },
	{ "PUSH13", Instruction::PUSH13 },
	{ "PUSH14", Instruction::PUSH14 },
	{ "PUSH15", Instruction::PUSH15 },
	{ "PUSH16", Instruction::PUSH16 },
	{ "PUSH17", Instruction::PUSH17 },
	{ "PUSH18", Instruction::PUSH18 },
	{ "PUSH19", Instruction::PUSH19 },
	{ "PUSH20", Instruction::PUSH20 },
	{ "PUSH21", Instruction::PUSH21 },
	{ "PUSH22", Instruction::PUSH22 },
	{ "PUSH23", Instruction::PUSH23 },
	{ "PUSH24", Instruction::PUSH24 },
	{ "PUSH25", Instruction::PUSH25 },
	{ "PUSH26", Instruction::PUSH26 },
	{ "PUSH27", Instruction::PUSH27 },
	{ "PUSH28", Instruction::PUSH28 },
	{ "PUSH29", Instruction::PUSH29 },
	{ "PUSH30", Instruction::PUSH30 },
	{ "PUSH31", Instruction::PUSH31 },
	{ "PUSH32", Instruction::PUSH32 },
	{ "CREATE", Instruction::CREATE },
	{ "CALL", Instruction::CALL },
	{ "RETURN", Instruction::RETURN },
	{ "SUICIDE", Instruction::SUICIDE }
};

const std::map<Instruction, InstructionInfo> eth::c_instructionInfo =
{ //                                           Add, Args, Ret
	{ Instruction::STOP,         { "STOP",         0, 0, 0 } },
	{ Instruction::ADD,          { "ADD",          0, 2, 1 } },
	{ Instruction::SUB,          { "SUB",          0, 2, 1 } },
	{ Instruction::MUL,          { "MUL",          0, 2, 1 } },
	{ Instruction::DIV,          { "DIV",          0, 2, 1 } },
	{ Instruction::SDIV,         { "SDIV",         0, 2, 1 } },
	{ Instruction::MOD,          { "MOD",          0, 2, 1 } },
	{ Instruction::SMOD,         { "SMOD",         0, 2, 1 } },
	{ Instruction::EXP,          { "EXP",          0, 2, 1 } },
	{ Instruction::NEG,          { "NEG",          0, 1, 1 } },
	{ Instruction::LT,           { "LT",           0, 2, 1 } },
	{ Instruction::GT,           { "GT",           0, 2, 1 } },
	{ Instruction::SLT,          { "SLT",          0, 2, 1 } },
	{ Instruction::SGT,          { "SGT",          0, 2, 1 } },
	{ Instruction::EQ,           { "EQ",           0, 2, 1 } },
	{ Instruction::NOT,          { "NOT",          0, 1, 1 } },
	{ Instruction::AND,          { "AND",          0, 2, 1 } },
	{ Instruction::OR,           { "OR",           0, 2, 1 } },
	{ Instruction::XOR,          { "XOR",          0, 2, 1 } },
	{ Instruction::BYTE,         { "BYTE",         0, 2, 1 } },
	{ Instruction::SHA3,         { "SHA3",         0, 2, 1 } },
	{ Instruction::ADDRESS,      { "ADDRESS",      0, 0, 1 } },
	{ Instruction::BALANCE,      { "BALANCE",      0, 1, 1 } },
	{ Instruction::ORIGIN,       { "ORIGIN",       0, 0, 1 } },
	{ Instruction::CALLER,       { "CALLER",       0, 0, 1 } },
	{ Instruction::CALLVALUE,    { "CALLVALUE",    0, 0, 1 } },
	{ Instruction::CALLDATALOAD, { "CALLDATALOAD", 0, 1, 1 } },
	{ Instruction::CALLDATASIZE, { "CALLDATASIZE", 0, 0, 1 } },
	{ Instruction::CALLDATACOPY, { "CALLDATACOPY", 0, 3, 0 } },
	{ Instruction::CODESIZE,     { "CODESIZE",     0, 0, 1 } },
	{ Instruction::CODECOPY,     { "CODECOPY",     0, 3, 0 } },
	{ Instruction::GASPRICE,     { "GASPRICE",     0, 0, 1 } },
	{ Instruction::PREVHASH,     { "PREVHASH",     0, 0, 1 } },
	{ Instruction::COINBASE,     { "COINBASE",     0, 0, 1 } },
	{ Instruction::TIMESTAMP,    { "TIMESTAMP",    0, 0, 1 } },
	{ Instruction::NUMBER,       { "NUMBER",       0, 0, 1 } },
	{ Instruction::DIFFICULTY,   { "DIFFICULTY",   0, 0, 1 } },
	{ Instruction::GASLIMIT,     { "GASLIMIT",     0, 0, 1 } },
	{ Instruction::POP,          { "POP",          0, 1, 0 } },
	{ Instruction::DUP,          { "DUP",          0, 1, 2 } },
	{ Instruction::SWAP,         { "SWAP",         0, 2, 2 } },
	{ Instruction::MLOAD,        { "MLOAD",        0, 1, 1 } },
	{ Instruction::MSTORE,       { "MSTORE",       0, 2, 0 } },
	{ Instruction::MSTORE8,      { "MSTORE8",      0, 2, 0 } },
	{ Instruction::SLOAD,        { "SLOAD",        0, 1, 1 } },
	{ Instruction::SSTORE,       { "SSTORE",       0, 2, 0 } },
	{ Instruction::JUMP,         { "JUMP",         0, 1, 0 } },
	{ Instruction::JUMPI,        { "JUMPI",        0, 2, 0 } },
	{ Instruction::PC,           { "PC",           0, 0, 1 } },
	{ Instruction::MEMSIZE,      { "MEMSIZE",      0, 0, 1 } },
	{ Instruction::GAS,          { "GAS",          0, 0, 1 } },
	{ Instruction::PUSH1,        { "PUSH1",        1, 0, 1 } },
	{ Instruction::PUSH2,        { "PUSH2",        2, 0, 1 } },
	{ Instruction::PUSH3,        { "PUSH3",        3, 0, 1 } },
	{ Instruction::PUSH4,        { "PUSH4",        4, 0, 1 } },
	{ Instruction::PUSH5,        { "PUSH5",        5, 0, 1 } },
	{ Instruction::PUSH6,        { "PUSH6",        6, 0, 1 } },
	{ Instruction::PUSH7,        { "PUSH7",        7, 0, 1 } },
	{ Instruction::PUSH8,        { "PUSH8",        8, 0, 1 } },
	{ Instruction::PUSH9,        { "PUSH9",        9, 0, 1 } },
	{ Instruction::PUSH10,       { "PUSH10",       10, 0, 1 } },
	{ Instruction::PUSH11,       { "PUSH11",       11, 0, 1 } },
	{ Instruction::PUSH12,       { "PUSH12",       12, 0, 1 } },
	{ Instruction::PUSH13,       { "PUSH13",       13, 0, 1 } },
	{ Instruction::PUSH14,       { "PUSH14",       14, 0, 1 } },
	{ Instruction::PUSH15,       { "PUSH15",       15, 0, 1 } },
	{ Instruction::PUSH16,       { "PUSH16",       16, 0, 1 } },
	{ Instruction::PUSH17,       { "PUSH17",       17, 0, 1 } },
	{ Instruction::PUSH18,       { "PUSH18",       18, 0, 1 } },
	{ Instruction::PUSH19,       { "PUSH19",       19, 0, 1 } },
	{ Instruction::PUSH20,       { "PUSH20",       20, 0, 1 } },
	{ Instruction::PUSH21,       { "PUSH21",       21, 0, 1 } },
	{ Instruction::PUSH22,       { "PUSH22",       22, 0, 1 } },
	{ Instruction::PUSH23,       { "PUSH23",       23, 0, 1 } },
	{ Instruction::PUSH24,       { "PUSH24",       24, 0, 1 } },
	{ Instruction::PUSH25,       { "PUSH25",       25, 0, 1 } },
	{ Instruction::PUSH26,       { "PUSH26",       26, 0, 1 } },
	{ Instruction::PUSH27,       { "PUSH27",       27, 0, 1 } },
	{ Instruction::PUSH28,       { "PUSH28",       28, 0, 1 } },
	{ Instruction::PUSH29,       { "PUSH29",       29, 0, 1 } },
	{ Instruction::PUSH30,       { "PUSH30",       30, 0, 1 } },
	{ Instruction::PUSH31,       { "PUSH31",       31, 0, 1 } },
	{ Instruction::PUSH32,       { "PUSH32",       32, 0, 1 } },
	{ Instruction::CREATE,       { "CREATE",       0, 3, 1 } },
	{ Instruction::CALL,         { "CALL",         0, 7, 1 } },
	{ Instruction::RETURN,       { "RETURN",       0, 2, 0 } },
	{ Instruction::SUICIDE,      { "SUICIDE",      0, 1, 0} }
};

void killBigints(sp::utree const& _this)
{
	switch (_this.which())
	{
	case sp::utree_type::list_type: for (auto const& i: _this) killBigints(i); break;
	case sp::utree_type::any_type: delete _this.get<bigint*>(); break;
	default:;
	}
}

void parseLLL(string const& _s, sp::utree& o_out)
{
	using qi::ascii::space;
	typedef sp::basic_string<std::string, sp::utree_type::symbol_type> symbol_type;
	typedef string::const_iterator it;

	qi::rule<it, qi::ascii::space_type, sp::utree()> element;
	qi::rule<it, string()> str = '"' > qi::lexeme[+(~qi::char_(std::string("\"") + '\0'))] > '"';
	qi::rule<it, string()> strsh = '\'' > qi::lexeme[+(~qi::char_(std::string(" ;())") + '\0'))];
	qi::rule<it, symbol_type()> symbol = qi::lexeme[+(~qi::char_(std::string(" @[]{}:();\"\x01-\x1f\x7f") + '\0'))];
	qi::rule<it, string()> intstr = qi::lexeme[ qi::no_case["0x"][qi::_val = "0x"] >> *qi::char_("0-9a-fA-F")[qi::_val += qi::_1]] | qi::lexeme[+qi::char_("0-9")[qi::_val += qi::_1]];
	qi::rule<it, bigint()> integer = intstr;
	qi::rule<it, bigint()> multiplier = qi::lit("wei")[qi::_val = 1] | qi::lit("szabo")[qi::_val = szabo] | qi::lit("finney")[qi::_val = finney] | qi::lit("ether")[qi::_val = ether];
	qi::rule<it, qi::ascii::space_type, bigint()> quantity = integer[qi::_val = qi::_1] >> -multiplier[qi::_val *= qi::_1];
	qi::rule<it, qi::ascii::space_type, sp::utree()> atom = quantity[qi::_val = px::construct<sp::any_ptr>(px::new_<bigint>(qi::_1))] | (str | strsh)[qi::_val = qi::_1] | symbol[qi::_val = qi::_1];
	qi::rule<it, qi::ascii::space_type, sp::utree::list_type()> seq = '{' > *element > '}';
	qi::rule<it, qi::ascii::space_type, sp::utree::list_type()> mload = '@' > element;
	qi::rule<it, qi::ascii::space_type, sp::utree::list_type()> sload = qi::lit("@@") > element;
	qi::rule<it, qi::ascii::space_type, sp::utree::list_type()> mstore = '[' > element > ']' > -qi::lit(":") > element;
	qi::rule<it, qi::ascii::space_type, sp::utree::list_type()> sstore = qi::lit("[[") > element > qi::lit("]]") > -qi::lit(":") > element;
	qi::rule<it, qi::ascii::space_type, sp::utree()> extra = sload[qi::_val = qi::_1, bind(&sp::utree::tag, qi::_val, 2)] | mload[qi::_val = qi::_1, bind(&sp::utree::tag, qi::_val, 1)] | sstore[qi::_val = qi::_1, bind(&sp::utree::tag, qi::_val, 4)] | mstore[qi::_val = qi::_1, bind(&sp::utree::tag, qi::_val, 3)] | seq[qi::_val = qi::_1, bind(&sp::utree::tag, qi::_val, 5)];
	qi::rule<it, qi::ascii::space_type, sp::utree::list_type()> list = '(' > *element > ')';
	element = atom | list | extra;

	try
	{
		string s;
		s.reserve(_s.size());
		bool incomment = false;
		bool instring = false;
		bool insstring = false;
		for (auto i: _s)
		{
			if (i == ';' && !instring && !insstring)
				incomment = true;
			else if (i == '\n')
				incomment = instring = insstring = false;
			else if (i == '"' && !insstring)
				instring = !instring;
			else if (i == '\'')
				insstring = true;
			else if (i == ' ')
				insstring = false;
			if (!incomment)
				s.push_back(i);
		}
		qi::phrase_parse(s.cbegin(), s.cend(), element, space, o_out);
	}
	catch (std::exception& _e)
	{
		cnote << _e.what();
	}
}

namespace eth
{

struct Macro
{
	std::vector<std::string> args;
	sp::utree code;
	std::map<std::string, CodeFragment> env;
};

static const CodeFragment NullCodeFragment;

struct CompilerState
{
	CodeFragment const& getDef(std::string const& _s)
	{
		if (defs.count(_s))
			return defs.at(_s);
		else if (args.count(_s))
			return args.at(_s);
		else if (outers.count(_s))
			return outers.at(_s);
		else
			return NullCodeFragment;
	}

	std::map<std::string, unsigned> vars;
	std::map<std::string, CodeFragment> defs;
	std::map<std::string, CodeFragment> args;
	std::map<std::string, CodeFragment> outers;
	std::map<std::string, Macro> macros;
	std::vector<sp::utree> treesToKill;
};

}

CodeLocation::CodeLocation(CodeFragment* _f)
{
	m_f = _f;
	m_pos = _f->m_code.size();
}

unsigned CodeLocation::get() const
{
	assert(m_f->m_code[m_pos - 1] == (byte)Instruction::PUSH4);
	bytesConstRef r(&m_f->m_code[m_pos], 4);
	cdebug << toHex(r);
	return fromBigEndian<uint32_t>(r);
}

void CodeLocation::set(unsigned _val)
{
	assert(m_f->m_code[m_pos - 1] == (byte)Instruction::PUSH4);
	assert(!get());
	bytesRef r(&m_f->m_code[m_pos], 4);
	toBigEndian(_val, r);
}

void CodeLocation::anchor()
{
	set(m_f->m_code.size());
}

void CodeLocation::increase(unsigned _val)
{
	assert(m_f->m_code[m_pos - 1] == (byte)Instruction::PUSH4);
	bytesRef r(&m_f->m_code[m_pos], 4);
	toBigEndian(get() + _val, r);
}

void CodeFragment::appendFragment(CodeFragment const& _f)
{
	m_locs.reserve(m_locs.size() + _f.m_locs.size());
	m_code.reserve(m_code.size() + _f.m_code.size());

	unsigned os = m_code.size();

	for (auto i: _f.m_code)
		m_code.push_back(i);

	for (auto i: _f.m_locs)
	{
		CodeLocation(this, i + os).increase(os);
		m_locs.push_back(i + os);
	}

	for (auto i: _f.m_data)
		m_data.insert(make_pair(i.first, i.second + os));

	m_deposit += _f.m_deposit;
}

void CodeFragment::consolidateData()
{
	m_code.push_back(0);
	bytes ld;
	for (auto const& i: m_data)
	{
		if (ld != i.first)
		{
			ld = i.first;
			for (auto j: ld)
				m_code.push_back(j);
		}
		CodeLocation(this, i.second).set(m_code.size() - ld.size());
	}
	m_data.clear();
}

void CodeFragment::appendFragment(CodeFragment const& _f, unsigned _deposit)
{
	if ((int)_deposit > _f.m_deposit)
		error<InvalidDeposit>();
	else
	{
		appendFragment(_f);
		while (_deposit++ < (unsigned)_f.m_deposit)
			appendInstruction(Instruction::POP);
	}
}

CodeLocation CodeFragment::appendPushLocation(unsigned _locationValue)
{
	m_code.push_back((byte)Instruction::PUSH4);
	CodeLocation ret(this, m_code.size());
	m_locs.push_back(m_code.size());
	m_code.resize(m_code.size() + 4);
	bytesRef r(&m_code[m_code.size() - 4], 4);
	toBigEndian(_locationValue, r);
	m_deposit++;
	return ret;
}

unsigned CodeFragment::appendPush(u256 _literalValue)
{
	unsigned br = max<unsigned>(1, bytesRequired(_literalValue));
	m_code.push_back((byte)Instruction::PUSH1 + br - 1);
	m_code.resize(m_code.size() + br);
	for (unsigned i = 0; i < br; ++i)
	{
		m_code[m_code.size() - 1 - i] = (byte)(_literalValue & 0xff);
		_literalValue >>= 8;
	}
	m_deposit++;
	return br + 1;
}

void CodeFragment::appendInstruction(Instruction _i)
{
	m_code.push_back((byte)_i);
	m_deposit += c_instructionInfo.at(_i).ret - c_instructionInfo.at(_i).args;
}

void debugOutAST(ostream& _out, sp::utree const& _this)
{
	switch (_this.which())
	{
	case sp::utree_type::list_type:
		switch (_this.tag())
		{
		case 0: _out << "( "; for (auto const& i: _this) { debugOutAST(_out, i); _out << " "; } _out << ")"; break;
		case 1: _out << "@ "; debugOutAST(_out, _this.front()); break;
		case 2: _out << "@@ "; debugOutAST(_out, _this.front()); break;
		case 3: _out << "[ "; debugOutAST(_out, _this.front()); _out << " ] "; debugOutAST(_out, _this.back()); break;
		case 4: _out << "[[ "; debugOutAST(_out, _this.front()); _out << " ]] "; debugOutAST(_out, _this.back()); break;
		case 5: _out << "{ "; for (auto const& i: _this) { debugOutAST(_out, i); _out << " "; } _out << "}"; break;
		default:;
		}

		break;
	case sp::utree_type::int_type: _out << _this.get<int>(); break;
	case sp::utree_type::string_type: _out << "\"" << _this.get<sp::basic_string<boost::iterator_range<char const*>, sp::utree_type::string_type>>() << "\""; break;
	case sp::utree_type::symbol_type: _out << _this.get<sp::basic_string<boost::iterator_range<char const*>, sp::utree_type::symbol_type>>(); break;
	case sp::utree_type::any_type: _out << *_this.get<bigint*>(); break;
	default: _out << "nil";
	}
}

CodeFragment::CodeFragment(sp::utree const& _t, CompilerState& _s, bool _allowASM)
{
	cdebug << "CodeFragment. Locals:";
	for (auto const& i: _s.defs)
		cdebug << i.first << ":" << toHex(i.second.m_code);
	cdebug << "Args:";
	for (auto const& i: _s.args)
		cdebug << i.first << ":" << toHex(i.second.m_code);
	cdebug << "Outers:";
	for (auto const& i: _s.outers)
		cdebug << i.first << ":" << toHex(i.second.m_code);
	debugOutAST(cout, _t);
	cout << endl << flush;

	switch (_t.which())
	{
	case sp::utree_type::list_type:
		constructOperation(_t, _s);
		break;
	case sp::utree_type::string_type:
	{
		auto sr = _t.get<sp::basic_string<boost::iterator_range<char const*>, sp::utree_type::string_type>>();
		string s(sr.begin(), sr.end());
		if (s.size() > 32)
			error<StringTooLong>();
		h256 valHash;
		memcpy(valHash.data(), s.data(), s.size());
		memset(valHash.data() + s.size(), 0, 32 - s.size());
		appendPush(valHash);
		break;
	}
	case sp::utree_type::symbol_type:
	{
		auto sr = _t.get<sp::basic_string<boost::iterator_range<char const*>, sp::utree_type::symbol_type>>();
		string s(sr.begin(), sr.end());
		string us = boost::algorithm::to_upper_copy(s);
		if (_allowASM)
		{
			if (c_instructions.count(us))
			{
				auto it = c_instructions.find(us);
				m_deposit = c_instructionInfo.at(it->second).ret - c_instructionInfo.at(it->second).args;
				m_code.push_back((byte)it->second);
			}
		}
		if (_s.defs.count(s))
			appendFragment(_s.defs.at(s));
		else if (_s.args.count(s))
			appendFragment(_s.args.at(s));
		else if (_s.outers.count(s))
			appendFragment(_s.outers.at(s));
		else if (us.find_first_of("1234567890") != 0 && us.find_first_not_of("QWERTYUIOPASDFGHJKLZXCVBNM1234567890") == string::npos)
		{
			auto it = _s.vars.find(s);
			if (it == _s.vars.end())
			{
				bool ok;
				tie(it, ok) = _s.vars.insert(make_pair(s, _s.vars.size() * 32));
			}
			appendPush(it->second);
		}
		else
			error<BareSymbol>();

		break;
	}
	case sp::utree_type::any_type:
	{
		bigint i = *_t.get<bigint*>();
		if (i < 0 || i > bigint(u256(0) - 1))
			error<IntegerOutOfRange>();
		appendPush((u256)i);
		break;
	}
	default: break;
	}
}

void CodeFragment::appendPushDataLocation(bytes const& _data)
{
	m_code.push_back((byte)Instruction::PUSH4);
	m_data.insert(make_pair(_data, m_code.size()));
	m_code.resize(m_code.size() + 4);
	memset(&m_code.back() - 3, 0, 4);
	m_deposit++;
}

std::string CodeFragment::asPushedString() const
{
	string ret;
	if (m_code.size())
	{
		unsigned bc = m_code[0] - (byte)Instruction::PUSH1 + 1;
		if (m_code[0] >= (byte)Instruction::PUSH1 && m_code[0] <= (byte)Instruction::PUSH32)
		{
			for (unsigned s = 0; s < bc && m_code[1 + s]; ++s)
				ret.push_back(m_code[1 + s]);
			return ret;
		}
	}
	error<ExpectedLiteral>();
	return ret;
}

CodeFragment compileLLLFragment(string const& _src, CompilerState& _s)
{
	CodeFragment ret;
	sp::utree o;
	parseLLL(_src, o);
	debugOutAST(cerr, o);
	cerr << endl;
	if (!o.empty())
		ret = CodeFragment(o, _s);
	_s.treesToKill.push_back(o);
	return ret;
}

void CodeFragment::constructOperation(sp::utree const& _t, CompilerState& _s)
{
	if (_t.empty())
		error<EmptyList>();
	else if (_t.tag() == 0 && _t.front().which() != sp::utree_type::symbol_type)
		error<DataNotExecutable>();
	else
	{
		string s;
		string us;
		switch (_t.tag())
		{
		case 0:
		{
			auto sr = _t.front().get<sp::basic_string<boost::iterator_range<char const*>, sp::utree_type::symbol_type>>();
			s = string(sr.begin(), sr.end());
			us = boost::algorithm::to_upper_copy(s);
			break;
		}
		case 1:
			us = "MLOAD";
			break;
		case 2:
			us = "SLOAD";
			break;
		case 3:
			us = "MSTORE";
			break;
		case 4:
			us = "SSTORE";
			break;
		case 5:
			us = "SEQ";
			break;
		default:;
		}

		// Operations who args are not standard stack-pushers.
		bool nonStandard = true;
		if (us == "ASM")
		{
			int c = 0;
			for (auto const& i: _t)
				if (c++)
					appendFragment(CodeFragment(i, _s, true));
		}
		else if (us == "INCLUDE")
		{
			if (_t.size() != 2)
				error<IncorrectParameterCount>();
			string n;
			auto i = *++_t.begin();
			if (i.tag())
				error<InvalidName>();
			if (i.which() == sp::utree_type::string_type)
			{
				auto sr = i.get<sp::basic_string<boost::iterator_range<char const*>, sp::utree_type::string_type>>();
				n = string(sr.begin(), sr.end());
			}
			else if (i.which() == sp::utree_type::symbol_type)
			{
				auto sr = i.get<sp::basic_string<boost::iterator_range<char const*>, sp::utree_type::symbol_type>>();
				n = _s.getDef(string(sr.begin(), sr.end())).asPushedString();
			}
			appendFragment(compileLLLFragment(asString(contents(n)), _s));
		}
		else if (us == "DEF")
		{
			string n;
			unsigned ii = 0;
			if (_t.size() != 3 && _t.size() != 4)
				error<IncorrectParameterCount>();
			for (auto const& i: _t)
			{
				if (ii == 1)
				{
					if (i.tag())
						error<InvalidName>();
					if (i.which() == sp::utree_type::string_type)
					{
						auto sr = i.get<sp::basic_string<boost::iterator_range<char const*>, sp::utree_type::string_type>>();
						n = string(sr.begin(), sr.end());
					}
					else if (i.which() == sp::utree_type::symbol_type)
					{
						auto sr = i.get<sp::basic_string<boost::iterator_range<char const*>, sp::utree_type::symbol_type>>();
						n = _s.getDef(string(sr.begin(), sr.end())).asPushedString();
					}
				}
				else if (ii == 2)
					if (_t.size() == 3)
						_s.defs[n] = CodeFragment(i, _s);
					else
						for (auto const& j: i)
						{
							if (j.tag() || j.which() != sp::utree_type::symbol_type)
								error<InvalidMacroArgs>();
							auto sr = j.get<sp::basic_string<boost::iterator_range<char const*>, sp::utree_type::symbol_type>>();
							_s.macros[n].args.push_back(string(sr.begin(), sr.end()));
						}
				else if (ii == 3)
				{
					_s.macros[n].code = i;
					_s.macros[n].env = _s.outers;
					for (auto const& i: _s.args)
						_s.macros[n].env[i.first] = i.second;
					for (auto const& i: _s.defs)
						_s.macros[n].env[i.first] = i.second;
				}
				++ii;
			}

		}
		else if (us == "LIT")
		{
			if (_t.size() < 3)
				error<IncorrectParameterCount>();
			unsigned ii = 0;
			CodeFragment pos;
			bytes data;
			for (auto const& i: _t)
			{
				if (ii == 1)
				{
					pos = CodeFragment(i, _s);
					if (pos.m_deposit != 1)
						error<InvalidDeposit>();
				}
				else if (ii == 2 && !i.tag() && i.which() == sp::utree_type::string_type)
				{
					auto sr = i.get<sp::basic_string<boost::iterator_range<char const*>, sp::utree_type::string_type>>();
					data = bytes((byte const*)sr.begin(), (byte const*)sr.end());
				}
				else if (ii >= 2 && !i.tag() && i.which() == sp::utree_type::any_type)
				{
					bigint bi = *i.get<bigint*>();
					if (bi < 0)
						error<IntegerOutOfRange>();
					else if (bi > bigint(u256(0) - 1))
					{
						if (ii == 2 && _t.size() == 3)
						{
							// One big int - allow it as hex.
							data.resize(bytesRequired(bi));
							toBigEndian(bi, data);
						}
						else
							error<IntegerOutOfRange>();
					}
					else
					{
						data.resize(data.size() + 32);
						*(h256*)(&data.back() - 31) = (u256)bi;
					}
				}
				else if (ii)
					error<InvalidLiteral>();
				++ii;
			}
			appendPush(data.size());
			appendInstruction(Instruction::DUP);
			appendPushDataLocation(data);
			appendFragment(pos, 1);
			appendInstruction(Instruction::CODECOPY);
		}
		else
			nonStandard = false;

		if (nonStandard)
			return;

		std::map<std::string, Instruction> const c_arith = { { "+", Instruction::ADD }, { "-", Instruction::SUB }, { "*", Instruction::MUL }, { "/", Instruction::DIV }, { "%", Instruction::MOD }, { "&", Instruction::AND }, { "|", Instruction::OR }, { "^", Instruction::XOR } };
		std::map<std::string, pair<Instruction, bool>> const c_binary = { { "<", { Instruction::LT, false } }, { "<=", { Instruction::GT, true } }, { ">", { Instruction::GT, false } }, { ">=", { Instruction::LT, true } }, { "S<", { Instruction::SLT, false } }, { "S<=", { Instruction::SGT, true } }, { "S>", { Instruction::SGT, false } }, { "S>=", { Instruction::SLT, true } }, { "=", { Instruction::EQ, false } }, { "!=", { Instruction::EQ, true } } };
		std::map<std::string, Instruction> const c_unary = { { "!", Instruction::NOT } };

		vector<CodeFragment> code;
		CompilerState ns = _s;
		ns.vars.clear();
		int c = _t.tag() ? 1 : 0;
		for (auto const& i: _t)
			if (c++)
			{
				if (us == "LLL" && c == 1)
					code.push_back(CodeFragment(i, ns));
				else
					code.push_back(CodeFragment(i, _s));
			}
		auto requireSize = [&](unsigned s) { if (code.size() != s) error<IncorrectParameterCount>(); };
		auto requireMinSize = [&](unsigned s) { if (code.size() < s) error<IncorrectParameterCount>(); };
		auto requireMaxSize = [&](unsigned s) { if (code.size() > s) error<IncorrectParameterCount>(); };
		auto requireDeposit = [&](unsigned i, int s) { if (code[i].m_deposit != s) error<InvalidDeposit>(); };

		if (_s.macros.count(s) && _s.macros.at(s).args.size() == code.size())
		{
			Macro const& m = _s.macros.at(s);
			CompilerState cs = _s;
			for (auto const& i: m.env)
				cs.outers[i.first] = i.second;
			for (auto const& i: cs.defs)
				cs.outers[i.first] = i.second;
			cs.defs.clear();
			for (unsigned i = 0; i < m.args.size(); ++i)
			{
				requireDeposit(i, 1);
				cs.args[m.args[i]] = code[i];
			}
			appendFragment(CodeFragment(m.code, cs));
			for (auto const& i: cs.defs)
				_s.defs[i.first] = i.second;
			for (auto const& i: cs.macros)
				_s.macros.insert(i);
		}
		else if (c_instructions.count(us))
		{
			auto it = c_instructions.find(us);
			int ea = c_instructionInfo.at(it->second).args;
			if (ea >= 0)
				requireSize(ea);
			else
				requireMinSize(-ea);

			for (unsigned i = code.size(); i; --i)
				appendFragment(code[i - 1], 1);
			appendInstruction(it->second);
		}
		else if (c_arith.count(us))
		{
			auto it = c_arith.find(us);
			requireMinSize(1);
			for (unsigned i = code.size(); i; --i)
			{
				requireDeposit(i - 1, 1);
				appendFragment(code[i - 1], 1);
			}
			for (unsigned i = 1; i < code.size(); ++i)
				appendInstruction(it->second);
		}
		else if (c_binary.count(us))
		{
			auto it = c_binary.find(us);
			requireSize(2);
			requireDeposit(0, 1);
			requireDeposit(1, 1);
			appendFragment(code[1], 1);
			appendFragment(code[0], 1);
			appendInstruction(it->second.first);
			if (it->second.second)
				appendInstruction(Instruction::NOT);
		}
		else if (c_unary.count(us))
		{
			auto it = c_unary.find(us);
			requireSize(1);
			requireDeposit(0, 1);
			appendFragment(code[0], 1);
			appendInstruction(it->second);
		}
		else if (us == "IF")
		{
			requireSize(3);
			requireDeposit(0, 1);
			appendFragment(code[0]);
			auto pos = appendJumpI();
			onePath();
			appendFragment(code[2]);
			auto end = appendJump();
			otherPath();
			pos.anchor();
			appendFragment(code[1]);
			donePaths();
			end.anchor();
		}
		else if (us == "WHEN" || us == "UNLESS")
		{
			requireSize(2);
			requireDeposit(0, 1);
			appendFragment(code[0]);
			if (us == "WHEN")
				appendInstruction(Instruction::NOT);
			auto end = appendJumpI();
			onePath();
			otherPath();
			appendFragment(code[1], 0);
			donePaths();
			end.anchor();
		}
		else if (us == "WHILE")
		{
			requireSize(2);
			requireDeposit(0, 1);
			auto begin = CodeLocation(this);
			appendFragment(code[0], 1);
			appendInstruction(Instruction::NOT);
			auto end = appendJumpI();
			appendFragment(code[1], 0);
			appendJump(begin);
			end.anchor();
		}
		else if (us == "FOR")
		{
			requireSize(4);
			requireDeposit(1, 1);
			appendFragment(code[0], 0);
			auto begin = CodeLocation(this);
			appendFragment(code[1], 1);
			appendInstruction(Instruction::NOT);
			auto end = appendJumpI();
			appendFragment(code[3], 0);
			appendFragment(code[2], 0);
			appendJump(begin);
			end.anchor();
		}
		else if (us == "LLL")
		{
			requireMinSize(2);
			requireMaxSize(3);
			requireDeposit(1, 1);

			CodeLocation codeloc(this, m_code.size() + 6);
			bytes const& subcode = code[0].code();
			appendPush(subcode.size());
			appendInstruction(Instruction::DUP);
			if (code.size() == 3)
			{
				requireDeposit(2, 1);
				appendFragment(code[2], 1);
				appendInstruction(Instruction::LT);
				appendInstruction(Instruction::NOT);
				appendInstruction(Instruction::MUL);
				appendInstruction(Instruction::DUP);
			}
			appendPushDataLocation(subcode);
			appendFragment(code[1], 1);
			appendInstruction(Instruction::CODECOPY);
		}
		else if (us == "&&" || us == "||")
		{
			requireMinSize(1);
			for (unsigned i = 0; i < code.size(); ++i)
				requireDeposit(i, 1);

			vector<CodeLocation> ends;
			if (code.size() > 1)
			{
				appendPush(us == "||" ? 1 : 0);
				for (unsigned i = 1; i < code.size(); ++i)
				{
					// Check if true - predicate
					appendFragment(code[i - 1], 1);
					if (us == "&&")
						appendInstruction(Instruction::NOT);
					ends.push_back(appendJumpI());
				}
				appendInstruction(Instruction::POP);
			}

			// Check if true - predicate
			appendFragment(code.back(), 1);

			// At end now.
			for (auto& i: ends)
				i.anchor();
		}
		else if (us == "~")
		{
			requireSize(1);
			requireDeposit(0, 1);
			appendFragment(code[0], 1);
			appendPush(1);
			appendPush(0);
			appendInstruction(Instruction::SUB);
			appendInstruction(Instruction::SUB);
		}
		else if (us == "SEQ")
		{
			unsigned ii = 0;
			for (auto const& i: code)
				if (++ii < code.size())
					appendFragment(i, 0);
				else
					appendFragment(i);
		}
		else if (us.find_first_of("1234567890") != 0 && us.find_first_not_of("QWERTYUIOPASDFGHJKLZXCVBNM1234567890") == string::npos)
		{
			auto it = _s.vars.find(s);
			if (it == _s.vars.end())
			{
				bool ok;
				tie(it, ok) = _s.vars.insert(make_pair(s, _s.vars.size() * 32));
			}
			appendPush(it->second);
		}
		else
			error<InvalidOperation>();
	}
}

bytes eth::compileLLL(string const& _s, vector<string>* _errors)
{
	try
	{
		CompilerState cs;
		bytes ret = compileLLLFragment(_s, cs).code();
		for (auto i: cs.treesToKill)
			killBigints(i);
		return ret;
	}
	catch (Exception const& _e)
	{
		if (_errors)
			_errors->push_back(_e.description());
	}
	catch (std::exception)
	{
		if (_errors)
			_errors->push_back("Parse error.");
	}
	return bytes();
}

string eth::disassemble(bytes const& _mem)
{
	stringstream ret;
	uint numerics = 0;
	for (auto it = _mem.begin(); it != _mem.end(); ++it)
	{
		byte n = *it;
		auto iit = c_instructionInfo.find((Instruction)n);
		if (numerics || iit == c_instructionInfo.end() || (byte)iit->first != n)	// not an instruction or expecting an argument...
		{
			if (numerics)
				numerics--;
			ret << "0x" << hex << (int)n << " ";
		}
		else
		{
			auto const& ii = iit->second;
			ret << ii.name << " ";
			numerics = ii.additional;
		}
	}
	return ret.str();
}
