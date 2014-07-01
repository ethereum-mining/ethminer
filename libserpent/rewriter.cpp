#include <stdio.h>
#include <iostream>
#include <vector>
#include <map>
#include "util.h"
#include "lllparser.h"
#include "bignum.h"

std::string valid[][3] = {
    { "if", "2", "3" },
    { "unless", "2", "2" },
    { "while", "2", "2" },
    { "until", "2", "2" },
    { "code", "1", "2" },
    { "init", "2", "2" },
    { "shared", "2", "3" },
    { "alloc", "1", "1" },
    { "array", "1", "1" },
    { "call", "2", "4" },
    { "create", "1", "4" },
    { "msg", "4", "6" },
    { "getch", "2", "2" },
    { "setch", "3", "3" },
    { "sha3", "1", "2" },
    { "return", "1", "2" },
    { "inset", "1", "1" },
    { "array_lit", "0", tt256 },
    { "seq", "0", tt256 },
    { "---END---", "", "" } //Keep this line at the end of the list
};

std::string macros[][2] = {
    {
        "(+= $a $b)",
        "(set $a (+ $a $b))"
    },
    {
        "(*= $a $b)",
        "(set $a (* $a $b))"
    },
    {
        "(-= $a $b)",
        "(set $a (- $a $b))"
    },
    {
        "(/= $a $b)",
        "(set $a (/ $a $b))"
    },
    {
        "(%= $a $b)",
        "(set $a (% $a $b))"
    },
    {
        "(^= $a $b)",
        "(set $a (^ $a $b))"
    },
    {
        "(@/= $a $b)",
        "(set $a (@/ $a $b))"
    },
    {
        "(@%= $a $b)",
        "(set $a (@% $a $b))"
    },
    {
        "(!= $a $b)",
        "(not (eq $a $b))"
    },
    {
        "(if $cond $do (else $else))",
        "(if $cond $do $else)"
    },
    {
        "(code $code)",
        "$code"
    },
    {
        "(access msg.data $ind)",
        "(calldataload (mul 32 $ind))"
    },
    {
        "(array $len)",
        "(alloc (mul 32 $len))"
    },
    {
        "(while $cond $do)",
        "(until (not $cond) $do)",
    },
    {
        "(while (not $cond) $do)",
        "(until $cond $do)",
    },
    {
        "(if $cond $do)",
        "(unless (not $cond) $do)",
    },
    {
        "(if (not $cond) $do)",
        "(unless $cond $do)",
    },
    {
        "(access contract.storage $ind)",
        "(sload $ind)"
    },
    {
        "(access $var $ind)",
        "(mload (add $var (mul 32 $ind)))"
    },
    {
        "(set (access contract.storage $ind) $val)",
        "(sstore $ind $val)"
    },
    {
        "(set (access $var $ind) $val)",
        "(mstore (add $var (mul 32 $ind)) $val)"
    },
    {
        "(getch $var $ind)",
        "(mod (mload (add $var $ind)) 256)"
    },
    {
        "(setch $var $ind $val)",
        "(mstore8 (add $var $ind) $val)",
    },
    {
        "(send $to $value)",
        "(call (sub (gas) 25) $to $value 0 0 0 0)"
    },
    {
        "(send $gas $to $value)",
        "(call $gas $to $value 0 0 0 0)"
    },
    {
        "(sha3 $x)",
        "(seq (set $1 $x) (sha3 (ref $1) 32))"
    },
    {
        "(sha3 $mstart $msize)",
        "(~sha3 $mstart (mul 32 $msize))"
    },
    {
        "(id $0)",
        "$0"
    },
    {
        "(return $x)",
        "(seq (set $1 $x) (~return (ref $1) 32))"
    },
    {
        "(return $start $len)",
        "(~return $start (mul 32 $len))"
    },
    {
        "(&& $x $y)",
        "(if $x $y 0)"
    },
    {
        "(|| $x $y)",
        "(seq (set $1 $x) (if (get $1) (get $1) $y))"
    },
    {
        "(>= $x $y)",
        "(not (slt $x $y))"
    },
    {
        "(<= $x $y)",
        "(not (sgt $x $y))"
    },
    {
        "(@>= $x $y)",
        "(not (lt $x $y))"
    },
    {
        "(@<= $x $y)",
        "(not (gt $x $y))"
    },
    {
        "(create $code)",
        "(create 0 $code)"
    },
    {
        "(create $endowment $code)",
        "(seq (set $1 (msize)) (create $endowment (get $1) (lll (outer $code) (msize))))"
    },
    {
        "(call $f $dataval)",
        "(msg (sub (gas) 45) $f 0 $dataval)"
    },
    {
        "(call $f $inp $inpsz)",
        "(msg (sub (gas) 25) $f 0 $inp $inpsz)"
    },
    {
        "(call $f $inp $inpsz $outsz)",
        "(seq (set $1 $outsz) (set $2 (alloc (mul 32 (get $1)))) (pop (call (sub (gas) (add 25 (get $1))) $f 0 $inp (mul 32 $inpsz) (get $2) (mul 32 (get $1)))) (get $2))"
    },
    {
        "(msg $gas $to $val $inp $inpsz)",
        "(seq (call $gas $to $val $inp (mul 32 $inpsz) (ref $1) 32) (get $1))"
    },
    {
        "(msg $gas $to $val $dataval)",
        "(seq (set $1 $dataval) (call $gas $to $val (ref $1) 32 (ref $2) 32) (get $2))"
    },
    {
        "(msg $gas $to $val $inp $inpsz $outsz)",
        "(seq (set $1 (mul 32 $outsz)) (set $2 (alloc (get $1))) (pop (call $gas $to $val $inp (mul 32 $inpsz) (get $2) (get $1))) (get $2))"
    },
    {
        "(outer (init $init $code))",
        "(seq $init (~return 0 (lll $code 0)))"
    },
    {
        "(outer (shared $shared (init $init (code $code))))",
        "(seq $shared $init (~return 0 (lll (seq $shared $code) 0)))"
    },
    {
        "(outer $code)",
        "(~return 0 (lll $code 0))"
    },
    {
        "(seq (seq) $x)",
        "$x"
    },
    {
        "(inset $x)",
        "$x"
    },
    {
        "(create $x)",
        "(seq (set $1 (msize)) (create $val (get $1) (lll $code (get $1))))"
    },
    { "msg.datasize", "(div (calldatasize) 32)" },
    { "msg.sender", "(caller)" },
    { "msg.value", "(callvalue)" },
    { "tx.gasprice", "(gasprice)" },
    { "tx.origin", "(origin)" },
    { "tx.gas", "(gas)" },
    { "contract.balance", "(balance)" },
    { "contract.address", "(address)" },
    { "block.prevhash", "(prevhash)" },
    { "block.coinbase", "(coinbase)" },
    { "block.timestamp", "(timestamp)" },
    { "block.number", "(number)" },
    { "block.difficulty", "(difficulty)" },
    { "block.gaslimit", "(gaslimit)" },
    { "stop", "(stop)" },
    { "---END---", "" } //Keep this line at the end of the list
};

std::vector<std::vector<Node> > nodeMacros;

std::string synonyms[][2] = {
    { "or", "||" },
    { "and", "&&" },
    { "elif", "if" },
    { "!", "not" },
    { "string", "alloc" },
    { "+", "add" },
    { "-", "sub" },
    { "*", "mul" },
    { "/", "sdiv" },
    { "^", "exp" },
    { "**", "exp" },
    { "%", "smod" },
    { "@/", "div" },
    { "@%", "mod" },
    { "@<", "lt" },
    { "@>", "gt" },
    { "<", "slt" },
    { ">", "sgt" },
    { "=", "set" },
    { "==", "eq" },
    { "---END---", "" } //Keep this line at the end of the list
};

struct matchResult {
    bool success;
    std::map<std::string, Node> map;
};

// Returns two values. First, a boolean to determine whether the node matches
// the pattern, second, if the node does match then a map mapping variables
// in the pattern to nodes
matchResult match(Node p, Node n) {
    matchResult o;
    o.success = false;
    if (p.type == TOKEN) {
        if (p.val == n.val && n.type == TOKEN) o.success = true;
        else if (p.val[0] == '$') {
            o.success = true;
            o.map[p.val.substr(1)] = n;
        }
    }
    else if (n.type==TOKEN || p.val!=n.val || p.args.size()!=n.args.size()) {
    }
    else {
		for (unsigned i = 0; i < p.args.size(); i++) {
            matchResult oPrime = match(p.args[i], n.args[i]);
            if (!oPrime.success) {
                o.success = false;
                return o;
            }
            for (std::map<std::string, Node>::iterator it = oPrime.map.begin();
                 it != oPrime.map.end();
                 it++) {
                o.map[(*it).first] = (*it).second;
            }
        }
        o.success = true;
    }
    return o;
}

// Fills in the pattern with a dictionary mapping variable names to
// nodes (these dicts are generated by match). Match and subst together
// create a full pattern-matching engine. 
Node subst(Node pattern,
           std::map<std::string, Node> dict,
           std::string varflag,
           Metadata metadata) {
    if (pattern.type == TOKEN && pattern.val[0] == '$') {
        if (dict.count(pattern.val.substr(1))) {
            return dict[pattern.val.substr(1)];
        }
        else {
            return token(varflag + pattern.val.substr(1), metadata);
        }
    }
    else if (pattern.type == TOKEN) {
        return pattern;
    }
    else {
        std::vector<Node> args;
		for (unsigned i = 0; i < pattern.args.size(); i++) {
            args.push_back(subst(pattern.args[i], dict, varflag, metadata));
        }
        return astnode(pattern.val, args, metadata);
    }
}

// array_lit transform

Node array_lit_transform(Node node) {
    std::vector<Node> o1;
    o1.push_back(token(intToDecimal(node.args.size() * 32), node.metadata));
    std::vector<Node> o2;
    std::string symb = "_temp"+mkUniqueToken()+"_0";
    o2.push_back(token(symb, node.metadata));
    o2.push_back(astnode("alloc", o1, node.metadata));
    std::vector<Node> o3;
    o3.push_back(astnode("set", o2, node.metadata));
	for (unsigned i = 0; i < node.args.size(); i++) {
        // (mstore (add (get symb) i*32) v)
        std::vector<Node> o5;
        o5.push_back(token(symb, node.metadata));
        std::vector<Node> o6;
        o6.push_back(astnode("get", o5, node.metadata));
        o6.push_back(token(intToDecimal(i * 32), node.metadata));
        std::vector<Node> o7;
        o7.push_back(astnode("add", o6));
        o7.push_back(node.args[i]);
        o3.push_back(astnode("mstore", o7, node.metadata));
    }
    std::vector<Node> o8;
    o8.push_back(token(symb, node.metadata));
    o3.push_back(astnode("get", o8));
    return astnode("seq", o3, node.metadata);
}

// Recursively applies rewrite rules
Node apply_rules(Node node) {
    // If the rewrite rules have not yet been parsed, parse them
    if (!nodeMacros.size()) {
        for (int i = 0; i < 9999; i++) {
            std::vector<Node> o;
            if (macros[i][0] == "---END---") break;
            o.push_back(parseLLL(macros[i][0]));
            o.push_back(parseLLL(macros[i][1]));
            nodeMacros.push_back(o);
        }
    }
    // Main code
	unsigned pos = 0;
    std::string prefix = "_temp"+mkUniqueToken()+"_";
    while(1) {
        if (synonyms[pos][0] == "---END---") {
            break;
        }
        else if (node.type == ASTNODE && node.val == synonyms[pos][0]) {
            node.val = synonyms[pos][1];
        }
        pos++;
    }
    for (pos = 0; pos < nodeMacros.size(); pos++) {
        Node pattern = nodeMacros[pos][0];
        matchResult mr = match(pattern, node);
        if (mr.success) {
            Node pattern2 = nodeMacros[pos][1];
            node = subst(pattern2, mr.map, prefix, node.metadata);
        }
    }
    // Array_lit special instruction
    if (node.val == "array_lit")
        node = array_lit_transform(node);
    if (node.type == ASTNODE && node.val != "ref" && node.val != "get") {
		unsigned i = 0;
        if (node.val == "set") i = 1;
        for (i = i; i < node.args.size(); i++) {
            node.args[i] = apply_rules(node.args[i]);
        }
    }
    else if (node.type == TOKEN && !isNumberLike(node)) {
        std::vector<Node> args;
        args.push_back(node);
        node = astnode("get", args, node.metadata);
    }
    // This allows people to use ~x as a way of having functions with the same
    // name and arity as macros; the idea is that ~x is a "final" form, and 
    // should not be remacroed, but it is converted back at the end
    if (node.type == ASTNODE && node.val[0] == '~')
        node.val = node.val.substr(1);
    return node;
}

Node optimize(Node inp) {
    if (inp.type == TOKEN) return tryNumberize(inp);
	for (unsigned i = 0; i < inp.args.size(); i++) {
        inp.args[i] = optimize(inp.args[i]);
    }
    if (inp.args.size() == 2 
            && inp.args[0].type == TOKEN 
            && inp.args[1].type == TOKEN) {
      std::string o;
      if (inp.val == "add") {
          o = decimalMod(decimalAdd(inp.args[0].val, inp.args[1].val), tt256);
      }
      else if (inp.val == "sub") {
          if (decimalGt(inp.args[0].val, inp.args[1].val, true))
              o = decimalSub(inp.args[0].val, inp.args[1].val);
      }
      else if (inp.val == "mul") {
          o = decimalMod(decimalMul(inp.args[0].val, inp.args[1].val), tt256);
      }
      else if (inp.val == "div" && inp.args[1].val != "0") {
          o = decimalDiv(inp.args[0].val, inp.args[1].val);
      }
      else if (inp.val == "sdiv" && inp.args[1].val != "0"
            && decimalGt(tt255, inp.args[0].val)
            && decimalGt(tt255, inp.args[1].val)) {
          o = decimalDiv(inp.args[0].val, inp.args[1].val);
      }
      else if (inp.val == "mod" && inp.args[1].val != "0") {
          o = decimalMod(inp.args[0].val, inp.args[1].val);
      }
      else if (inp.val == "smod" && inp.args[1].val != "0"
            && decimalGt(tt255, inp.args[0].val)
            && decimalGt(tt255, inp.args[1].val)) {
          o = decimalMod(inp.args[0].val, inp.args[1].val);
      }    
      if (o.length()) return token(o, inp.metadata);
    }
    return inp;
}

Node validate(Node inp) {
    if (inp.type == ASTNODE) {
        int i = 0;
        while(valid[i][0] != "---END---") {
            if (inp.val == valid[i][0]) {
                if (decimalGt(valid[i][1], intToDecimal(inp.args.size()))) {
                    err("Too few arguments for "+inp.val, inp.metadata);   
                }
                if (decimalGt(intToDecimal(inp.args.size()), valid[i][2])) {
                    err("Too many arguments for "+inp.val, inp.metadata);   
                }
            }
            i++;
        }
    }
	for (unsigned i = 0; i < inp.args.size(); i++) validate(inp.args[i]);
    return inp;
}

Node preprocess(Node inp) {
    std::vector<Node> args;
    args.push_back(inp);
    return astnode("outer", args, inp.metadata);
}

Node rewrite(Node inp) {
    return optimize(apply_rules(validate(preprocess(inp))));
}

using namespace std;
