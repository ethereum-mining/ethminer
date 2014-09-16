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
 * Ethereum client.
 */
#if 0
#define BOOST_RESULT_OF_USE_DECLTYPE
#define BOOST_SPIRIT_USE_PHOENIX_V3
#include <boost/spirit/include/qi.hpp>
#include <boost/spirit/include/phoenix.hpp>
#include <boost/spirit/include/support_utree.hpp>
#endif
#include <libdevcore/Log.h>
#include <libdevcore/Common.h>
#include <libdevcore/CommonData.h>
#include <libdevcore/RLP.h>
#include <libp2p/All.h>
#include <libwhisper/WhisperPeer.h>
#if 0
#include <libevm/VM.h>
#include "BuildInfo.h"
#endif
using namespace std;
using namespace dev;
using namespace dev::eth;
using namespace dev::p2p;
using namespace dev::shh;
#if 0
#if 0
namespace qi = boost::spirit::qi;
namespace px = boost::phoenix;
namespace sp = boost::spirit;


class ASTSymbol: public string
{
public:
	ASTSymbol() {}
};

enum class ASTType
{
	Symbol,
	IntegerLiteral,
	StringLiteral,
	Call,
	Return,
	Operator,
	Compound
};

class ASTNode: public vector<ASTNode>
{
public:
	ASTNode() {}
	ASTNode(ASTSymbol const& _s): m_type(ASTType::Symbol), m_s(_s) {}
	ASTNode(string const& _s): m_type(ASTType::StringLiteral), m_s(_s) {}
	ASTNode(bigint const& _i): m_type(ASTType::IntegerLiteral), m_i(_i) {}
	ASTNode(ASTType _t): m_type(_t) {}

	ASTNode& operator=(ASTSymbol const& _s) { m_type = ASTType::Symbol; m_s = _s; return *this; }
	ASTNode& operator=(string const& _s) { m_type = ASTType::StringLiteral; m_s = _s; return *this; }
	ASTNode& operator=(bigint const& _i) { m_type = ASTType::IntegerLiteral; m_i = _i; return *this; }
	ASTNode& operator=(ASTType const& _s) { m_type = _s; return *this; }

	void debugOut(ostream& _out) const;

private:
	ASTType m_type;
	string m_s;
	bigint m_i;
};

void parseTree(string const& _s, ASTNode& o_out)
{
	using qi::standard::space;
	using qi::standard::space_type;
	typedef string::const_iterator it;

/*	static const u256 ether = u256(1000000000) * 1000000000;
	static const u256 finney = u256(1000000000) * 1000000;
	static const u256 szabo = u256(1000000000) * 1000;*/

	qi::rule<it, space_type, ASTNode()> element;
	qi::rule<it, space_type, ASTNode()> call;
	qi::rule<it, string()> str = '"' > qi::lexeme[+(~qi::char_(std::string("\"") + '\0'))] > '"';
	qi::rule<it, ASTSymbol()> symbol = qi::lexeme[+(~qi::char_(std::string(" $@[]{}:();\"\x01-\x1f\x7f") + '\0'))];
/*	qi::rule<it, string()> strsh = '\'' > qi::lexeme[+(~qi::char_(std::string(" ;$@()[]{}:\n\t") + '\0'))];
	qi::rule<it, string()> intstr = qi::lexeme[ qi::no_case["0x"][qi::_val = "0x"] >> *qi::char_("0-9a-fA-F")[qi::_val += qi::_1]] | qi::lexeme[+qi::char_("0-9")[qi::_val += qi::_1]];
	qi::rule<it, bigint()> integer = intstr;
	qi::rule<it, bigint()> multiplier = qi::lit("wei")[qi::_val = 1] | qi::lit("szabo")[qi::_val = szabo] | qi::lit("finney")[qi::_val = finney] | qi::lit("ether")[qi::_val = ether];
	qi::rule<it, space_type, bigint()> quantity = integer[qi::_val = qi::_1] >> -multiplier[qi::_val *= qi::_1];
	qi::rule<it, space_type, sp::utree()> atom = quantity[qi::_val = px::construct<sp::any_ptr>(px::new_<bigint>(qi::_1))] | (str | strsh)[qi::_val = qi::_1] | symbol[qi::_val = qi::_1];
	qi::rule<it, space_type, sp::utree::list_type()> seq = '{' > *element > '}';
	qi::rule<it, space_type, sp::utree::list_type()> mload = '@' > element;
	qi::rule<it, space_type, sp::utree::list_type()> sload = qi::lit("@@") > element;
	qi::rule<it, space_type, sp::utree::list_type()> mstore = '[' > element > ']' > -qi::lit(":") > element;
	qi::rule<it, space_type, sp::utree::list_type()> sstore = qi::lit("[[") > element > qi::lit("]]") > -qi::lit(":") > element;
	qi::rule<it, space_type, sp::utree::list_type()> calldataload = qi::lit("$") > element;
	qi::rule<it, space_type, sp::utree::list_type()> list = '(' > *element > ')';

	qi::rule<it, space_type, sp::utree()> extra = sload[tagNode<2>()] | mload[tagNode<1>()] | sstore[tagNode<4>()] | mstore[tagNode<3>()] | seq[tagNode<5>()] | calldataload[tagNode<6>()];*/
	qi::rule<it, space_type, ASTNode()> value = call[qi::_val = ASTType::Call] | str[qi::_val = qi::_1] | symbol[qi::_val = qi::_1];
	qi::rule<it, space_type, ASTNode()> compound = '{' > *element > '}';
	call = '(' > *value > ')'; //symbol > '(' > !(value > *(',' > value)) > ')';
	element = compound[qi::_val = ASTType::Compound] | value[qi::_val = qi::_1];

	auto ret = _s.cbegin();
	qi::phrase_parse(ret, _s.cend(), element, space, qi::skip_flag::dont_postskip, o_out);
	for (auto i = ret; i != _s.cend(); ++i)
		if (!isspace(*i))
			throw std::exception();
}

void ASTNode::debugOut(ostream& _out) const
{
	switch (m_type)
	{
	case ASTType::StringLiteral:
		_out << "\"" << m_s << "\"";
		break;
	case ASTType::Symbol:
		_out << m_s;
		break;
	case ASTType::Compound:
	{
		unsigned n = 0;
		_out << "{";
		for (auto const& i: *this)
		{
			i.debugOut(_out);
			_out << ";";
			++n;
		}
		_out << "}";
		break;
	}
	case ASTType::Call:
	{
		unsigned n = 0;
		for (auto const& i: *this)
		{
			i.debugOut(_out);
			if (n == 0)
				_out << "(";
			else if (n < size() - 1)
				_out << ",";
			if (n == size() - 1)
				_out << ")";
			++n;
		}
		break;
	}
	default:
		_out << "nil";
	}
}

int main(int, char**)
{
	ASTNode out;
	parseTree("{x}", out);
	out.debugOut(cout);
	cout << endl;
	return 0;
}
#endif

void killBigints(sp::utree const& _this)
{
	switch (_this.which())
	{
	case sp::utree_type::list_type: for (auto const& i: _this) killBigints(i); break;
	case sp::utree_type::any_type: delete _this.get<bigint*>(); break;
	default:;
	}
}

void debugOutAST(ostream& _out, sp::utree const& _this)
{
	switch (_this.which())
	{
	case sp::utree_type::list_type:
		switch (_this.tag())
		{
		case 0: { int n = 0; for (auto const& i: _this) { debugOutAST(_out, i); if (n++) _out << ", "; } break; }
		case 1: _out << "@ "; debugOutAST(_out, _this.front()); break;
		case 2: _out << "@@ "; debugOutAST(_out, _this.front()); break;
		case 3: _out << "[ "; debugOutAST(_out, _this.front()); _out << " ] "; debugOutAST(_out, _this.back()); break;
		case 4: _out << "[[ "; debugOutAST(_out, _this.front()); _out << " ]] "; debugOutAST(_out, _this.back()); break;
		case 5: _out << "{ "; for (auto const& i: _this) { debugOutAST(_out, i); _out << " "; } _out << "}"; break;
		case 6: _out << "$ "; debugOutAST(_out, _this.front()); break;
		default:
			{ _out << _this.tag() << ": "; int n = 0; for (auto const& i: _this) { debugOutAST(_out, i); if (n++) _out << ", "; } break; }
		}

		break;
	case sp::utree_type::int_type: _out << _this.get<int>(); break;
	case sp::utree_type::string_type: _out << "\"" << _this.get<sp::basic_string<boost::iterator_range<char const*>, sp::utree_type::string_type>>() << "\""; break;
	case sp::utree_type::symbol_type: _out << _this.get<sp::basic_string<boost::iterator_range<char const*>, sp::utree_type::symbol_type>>(); break;
	case sp::utree_type::any_type: _out << *_this.get<bigint*>(); break;
	default: _out << "nil";
	}
}

namespace dev {
namespace eth {
namespace parseTreeLLL_ {

template<unsigned N>
struct tagNode
{
	void operator()(sp::utree& n, qi::rule<string::const_iterator, qi::ascii::space_type, sp::utree()>::context_type& c) const
	{
		(boost::fusion::at_c<0>(c.attributes) = n).tag(N);
	}
};

}}}

void parseTree(string const& _s, sp::utree& o_out)
{
	using qi::standard::space;
	using qi::standard::space_type;
	using dev::eth::parseTreeLLL_::tagNode;
	typedef sp::basic_string<std::string, sp::utree_type::symbol_type> symbol_type;
	typedef string::const_iterator it;

	static const u256 ether = u256(1000000000) * 1000000000;
	static const u256 finney = u256(1000000000) * 1000000;
	static const u256 szabo = u256(1000000000) * 1000;
#if 0
	qi::rule<it, space_type, sp::utree()> element;
	qi::rule<it, space_type, sp::utree()> statement;
	qi::rule<it, string()> str = '"' > qi::lexeme[+(~qi::char_(std::string("\"") + '\0'))] > '"';
	qi::rule<it, string()> strsh = '\'' > qi::lexeme[+(~qi::char_(std::string(" ;$@()[]{}:\n\t") + '\0'))];
	qi::rule<it, symbol_type()> symbol = qi::lexeme[+(~qi::char_(std::string(" $@[]{}:();\"\x01-\x1f\x7f") + '\0'))];
	qi::rule<it, string()> intstr = qi::lexeme[ qi::no_case["0x"][qi::_val = "0x"] >> *qi::char_("0-9a-fA-F")[qi::_val += qi::_1]] | qi::lexeme[+qi::char_("0-9")[qi::_val += qi::_1]];
	qi::rule<it, bigint()> integer = intstr;
	qi::rule<it, bigint()> multiplier = qi::lit("wei")[qi::_val = 1] | qi::lit("szabo")[qi::_val = szabo] | qi::lit("finney")[qi::_val = finney] | qi::lit("ether")[qi::_val = ether];
	qi::rule<it, space_type, bigint()> quantity = integer[qi::_val = qi::_1] >> -multiplier[qi::_val *= qi::_1];
	qi::rule<it, space_type, sp::utree()> atom = quantity[qi::_val = px::construct<sp::any_ptr>(px::new_<bigint>(qi::_1))] | (str | strsh)[qi::_val = qi::_1] | symbol[qi::_val = qi::_1];
	qi::rule<it, space_type, sp::utree::list_type()> compound = '{' > *statement > '}';
/*	qi::rule<it, space_type, sp::utree::list_type()> mload = '@' > element;
	qi::rule<it, space_type, sp::utree::list_type()> sload = qi::lit("@@") > element;
	qi::rule<it, space_type, sp::utree::list_type()> mstore = '[' > element > ']' > -qi::lit(":") > element;
	qi::rule<it, space_type, sp::utree::list_type()> sstore = qi::lit("[[") > element > qi::lit("]]") > -qi::lit(":") > element;
	qi::rule<it, space_type, sp::utree::list_type()> calldataload = qi::lit("$") > element;*/
//	qi::rule<it, space_type, sp::utree::list_type()> args = '(' > (element % ',') > ')';

	qi::rule<it, space_type, sp::utree::list_type()> expression;
	qi::rule<it, space_type, sp::utree()> group = '(' >> expression[qi::_val = qi::_1] >> ')';
	qi::rule<it, space_type, sp::utree()> factor = atom | group;
	qi::rule<it, space_type, sp::utree()> mul = '*' >> factor;
	qi::rule<it, space_type, sp::utree()> div = '/' >> factor;
	qi::rule<it, space_type, sp::utree()> op = mul[tagNode<10>()] | div[tagNode<11>()];
	qi::rule<it, space_type, sp::utree::list_type()> term = factor >> !op;
	expression  = term >> !(('+' >> term) | ('-' >> term));

	//	qi::rule<it, space_type, sp::utree()> extra = sload[tagNode<2>()] | mload[tagNode<1>()] | sstore[tagNode<4>()] | mstore[tagNode<3>()] | calldataload[tagNode<6>()];
	statement = compound[tagNode<5>()] | (element > ';')[qi::_val = qi::_1];
	element %= expression;// | extra;
#endif
	qi::rule<it, symbol_type()> symbol = qi::lexeme[+(~qi::char_(std::string(" $@[]{}:();\"\x01-\x1f\x7f") + '\0'))];
	qi::rule<it, string()> intstr = qi::lexeme[ qi::no_case["0x"][qi::_val = "0x"] >> *qi::char_("0-9a-fA-F")[qi::_val += qi::_1]] | qi::lexeme[+qi::char_("0-9")[qi::_val += qi::_1]];
	qi::rule<it, bigint()> integer = intstr;
	qi::rule<it, sp::utree()> intnode = integer[qi::_val = px::construct<sp::any_ptr>(px::new_<bigint>(qi::_1))];
	qi::rule<it, space_type, sp::utree()> funcname = symbol;
	qi::rule<it, space_type, sp::utree()> statement;
	qi::rule<it, space_type, sp::utree::list_type()> call = funcname > '(' > funcname > ')';
	statement = call | intnode | symbol;

	auto ret = _s.cbegin();
	qi::phrase_parse(ret, _s.cend(), statement, space, qi::skip_flag::dont_postskip, o_out);
	for (auto i = ret; i != _s.cend(); ++i)
		if (!isspace(*i))
			throw std::exception();
}
#endif
int main(int argc, char** argv)
{
#if 0
	sp::utree out;
	parseTree("x(2)", out);
	debugOutAST(cout, out);
	killBigints(out);
	cout << endl;
#endif

	g_logVerbosity = 20;

	short listenPort = 30303;
	string remoteHost;
	short remotePort = 30303;

	for (int i = 1; i < argc; ++i)
	{
		string arg = argv[i];
		if (arg == "-l" && i + 1 < argc)
			listenPort = (short)atoi(argv[++i]);
		else if (arg == "-r" && i + 1 < argc)
			remoteHost = argv[++i];
		else if (arg == "-p" && i + 1 < argc)
			remotePort = (short)atoi(argv[++i]);
		else
			remoteHost = argv[i];
	}

	Host ph("Test", NetworkPreferences(listenPort, "", false, true));
	ph.registerCapability(new WhisperHost());
	auto wh = ph.cap<WhisperHost>();

	ph.start();

	if (!remoteHost.empty())
		ph.connect(remoteHost, remotePort);

	/// Only interested in the packet if the lowest bit is 1
	auto w = wh->installWatch(MessageFilter(std::vector<std::pair<bytes, bytes> >({{fromHex("0000000000000000000000000000000000000000000000000000000000000001"), fromHex("0000000000000000000000000000000000000000000000000000000000000001")}})));


	for (int i = 0; ; ++i)
	{
		wh->sendRaw(h256(u256(i * i)).asBytes(), h256(u256(i)).asBytes(), 1000);
		for (auto i: wh->checkWatch(w))
			cnote << "New message:" << (u256)h256(wh->message(i).payload);
	}

	return 0;
}
