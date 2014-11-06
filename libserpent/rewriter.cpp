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
    { "alloc", "1", "1" },
    { "array", "1", "1" },
    { "call", "2", tt256 },
    { "call_code", "2", tt256 },
    { "create", "1", "4" },
    { "getch", "2", "2" },
    { "setch", "3", "3" },
    { "sha3", "1", "2" },
    { "return", "1", "2" },
    { "inset", "1", "1" },
    { "min", "2", "2" },
    { "max", "2", "2" },
    { "array_lit", "0", tt256 },
    { "seq", "0", tt256 },
    { "log", "1", "6" },
    { "outer", "1", "1" },
    { "set", "2", "2" },
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
        "(iszero (eq $a $b))"
    },
    {
        "(min a b)",
        "(with $1 a (with $2 b (if (lt $1 $2) $1 $2)))"
    },
    {
        "(max a b)",
        "(with $1 a (with $2 b (if (lt $1 $2) $2 $1)))"
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
        "(access (. msg data) $ind)",
        "(calldataload (mul 32 $ind))"
    },
    {
        "(slice $arr $pos)",
        "(add $arr (mul 32 $pos))",
    },
    {
        "(array $len)",
        "(alloc (mul 32 $len))"
    },
    {
        "(while $cond $do)",
        "(until (iszero $cond) $do)",
    },
    {
        "(while (iszero $cond) $do)",
        "(until $cond $do)",
    },
    {
        "(if $cond $do)",
        "(unless (iszero $cond) $do)",
    },
    {
        "(if (iszero $cond) $do)",
        "(unless $cond $do)",
    },
    {
        "(access (. self storage) $ind)",
        "(sload $ind)"
    },
    {
        "(access $var $ind)",
        "(mload (add $var (mul 32 $ind)))"
    },
    {
        "(set (access (. self storage) $ind) $val)",
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
        "(~call (sub (gas) 25) $to $value 0 0 0 0)"
    },
    {
        "(send $gas $to $value)",
        "(~call $gas $to $value 0 0 0 0)"
    },
    {
        "(sha3 $x)",
        "(seq (set $1 $x) (~sha3 (ref $1) 32))"
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
        "(with $1 $x (if (get $1) (get $1) $y))"
    },
    {
        "(>= $x $y)",
        "(iszero (slt $x $y))"
    },
    {
        "(<= $x $y)",
        "(iszero (sgt $x $y))"
    },
    {
        "(@>= $x $y)",
        "(iszero (lt $x $y))"
    },
    {
        "(@<= $x $y)",
        "(iszero (gt $x $y))"
    },
    {
        "(create $code)",
        "(create 0 $code)"
    },
    {
        "(create $endowment $code)",
        "(with $1 (msize) (create $endowment (get $1) (lll (outer $code) (msize))))"
    },
    {
        "(sha256 $x)",
        "(seq (set $1 $x) (pop (~call 101 2 0 (ref $1) 32 (ref $2) 32)) (get $2))"
    },
    {
        "(sha256 $arr $sz)",
        "(seq (pop (~call 101 2 0 $arr (mul 32 $sz) (ref $2) 32)) (get $2))"
    },
    {
        "(ripemd160 $x)",
        "(seq (set $1 $x) (pop (~call 101 3 0 (ref $1) 32 (ref $2) 32)) (get $2))"
    },
    {
        "(ripemd160 $arr $sz)",
        "(seq (pop (~call 101 3 0 $arr (mul 32 $sz) (ref $2) 32)) (get $2))"
    },
    {
        "(ecrecover $h $v $r $s)",
        "(seq (declare $1) (declare $2) (declare $3) (declare $4) (set $1 $h) (set $2 $v) (set $3 $r) (set $4 $s) (pop (~call 101 1 0 (ref $1) 128 (ref $5) 32)) (get $5))"
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
        "(with $1 (msize) (create $val (get $1) (lll $code (get $1))))"
    },
    {
        "(with (= $var $val) $cond)",
        "(with $var $val $cond)"
    },
    {
        "(log $t1)",
        "(~log1 $t1 0 0)"
    },
    {
        "(log $t1 $t2)",
        "(~log2 $t1 $t2 0 0)"
    },
    {
        "(log $t1 $t2 $t3)",
        "(~log3 $t1 $t2 $t3 0 0)"
    },
    {
        "(log $t1 $t2 $t3 $t4)",
        "(~log4 $t1 $t2 $t3 $t4 0 0)"
    },
    { "(. msg datasize)", "(div (calldatasize) 32)" },
    { "(. msg sender)", "(caller)" },
    { "(. msg value)", "(callvalue)" },
    { "(. tx gasprice)", "(gasprice)" },
    { "(. tx origin)", "(origin)" },
    { "(. tx gas)", "(gas)" },
    { "(. $x balance)", "(balance $x)" },
    { "self", "(address)" },
    { "(. block prevhash)", "(prevhash)" },
    { "(. block coinbase)", "(coinbase)" },
    { "(. block timestamp)", "(timestamp)" },
    { "(. block number)", "(number)" },
    { "(. block difficulty)", "(difficulty)" },
    { "(. block gaslimit)", "(gaslimit)" },
    { "stop", "(stop)" },
    { "---END---", "" } //Keep this line at the end of the list
};

std::vector<std::vector<Node> > nodeMacros;

std::string synonyms[][2] = {
    { "or", "||" },
    { "and", "&&" },
    { "|", "~or" },
    { "&", "~and" },
    { "elif", "if" },
    { "!", "iszero" },
    { "~", "~not" },
    { "not", "iszero" },
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
    { ":", "kv" },
    { "---END---", "" } //Keep this line at the end of the list
};

std::string setters[][2] = {
    { "+=", "+" },
    { "-=", "-" },
    { "*=", "*" },
    { "/=", "/" },
    { "%=", "%" },
    { "^=", "^" },
    { "!=", "!" },
    { "---END---", "" } //Keep this line at the end of the list
};

// Match result storing object
struct matchResult {
    bool success;
    std::map<std::string, Node> map;
};

// Storage variable index storing object
struct svObj {
    std::map<std::string, std::string> offsets;
    std::map<std::string, int> indices;
    std::map<std::string, std::vector<std::string> > coefficients;
    std::map<std::string, bool> nonfinal;
    std::string globalOffset;
};

// Preprocessing result storing object
class preprocessAux {
    public:
        preprocessAux() {
            globalExterns = std::map<std::string, int>();
            localExterns = std::map<std::string, std::map<std::string, int> >();
            localExterns["self"] = std::map<std::string, int>();
        }
        std::map<std::string, int> globalExterns;
        std::map<std::string, std::map<std::string, int> > localExterns;
        svObj storageVars;
};

#define preprocessResult std::pair<Node, preprocessAux>

// Main pattern matching routine, for those patterns that can be expressed
// using our standard mini-language above
//
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
        // do nothing
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

// Processes mutable array literals

Node array_lit_transform(Node node) {
    Metadata m = node.metadata;
    std::vector<Node> o1;
    o1.push_back(token(unsignedToDecimal(node.args.size() * 32), m));
    std::vector<Node> o2;
    std::string symb = "_temp"+mkUniqueToken()+"_0";
    o2.push_back(token(symb, m));
    o2.push_back(astnode("alloc", o1, m));
    std::vector<Node> o3;
    o3.push_back(astnode("set", o2, m));
    for (unsigned i = 0; i < node.args.size(); i++) {
        std::vector<Node> o5;
        o5.push_back(token(symb, m));
        std::vector<Node> o6;
        o6.push_back(astnode("get", o5, m));
        o6.push_back(token(unsignedToDecimal(i * 32), m));
        std::vector<Node> o7;
        o7.push_back(astnode("add", o6));
        o7.push_back(node.args[i]);
        o3.push_back(astnode("mstore", o7, m));
    }
    std::vector<Node> o8;
    o8.push_back(token(symb, m));
    o3.push_back(astnode("get", o8));
    return astnode("seq", o3, m);
}

// Is the given node something of the form
// self.cow
// self.horse[0]
// self.a[6][7][self.storage[3]].chicken[9]
bool isNodeStorageVariable(Node node) {
    std::vector<Node> nodez;
    nodez.push_back(node);
    while (1) {
        if (nodez.back().type == TOKEN) return false;
        if (nodez.back().args.size() == 0) return false;
        if (nodez.back().val != "." && nodez.back().val != "access")
            return false;
        if (nodez.back().args[0].val == "self") return true;
        nodez.push_back(nodez.back().args[0]);
    }
}

Node optimize(Node inp);

Node apply_rules(preprocessResult pr);

// Convert:
// self.cow -> ["cow"]
// self.horse[0] -> ["horse", "0"]
// self.a[6][7][self.storage[3]].chicken[9] -> 
//     ["6", "7", (sload 3), "chicken", "9"]
std::vector<Node> listfyStorageAccess(Node node) {
    std::vector<Node> out;
    std::vector<Node> nodez;
    nodez.push_back(node);
    while (1) {
        if (nodez.back().type == TOKEN) {
            out.push_back(token("--" + nodez.back().val, node.metadata));
            std::vector<Node> outrev;
            for (int i = (signed)out.size() - 1; i >= 0; i--) {
                outrev.push_back(out[i]);
            }
            return outrev;
        }
        if (nodez.back().val == ".")
            nodez.back().args[1].val = "--" + nodez.back().args[1].val;
        if (nodez.back().args.size() == 0)
            err("Error parsing storage variable statement", node.metadata);
        if (nodez.back().args.size() == 1)
            out.push_back(token(tt256m1, node.metadata));
        else
            out.push_back(nodez.back().args[1]);
        nodez.push_back(nodez.back().args[0]);
    }
}

// Cool function for debug purposes (named cerrStringList to make
// all prints searchable via 'cerr')
void cerrStringList(std::vector<std::string> s, std::string suffix="") {
    for (unsigned i = 0; i < s.size(); i++) std::cerr << s[i] << " ";
    std::cerr << suffix << "\n";
}

// Populate an svObj with the arguments needed to determine
// the storage position of a node
svObj getStorageVars(svObj pre, Node node, std::string prefix="", int index=0) {
    Metadata m = node.metadata;
    if (!pre.globalOffset.size()) pre.globalOffset = "0";
    std::vector<Node> h;
    std::vector<std::string> coefficients;
    // Array accesses or atoms
    if (node.val == "access" || node.type == TOKEN) {
        std::string tot = "1";
        h = listfyStorageAccess(node);
        coefficients.push_back("1");
        for (unsigned i = h.size() - 1; i >= 1; i--) {
            // Array sizes must be constant or at least arithmetically
            // evaluable at compile time
            h[i] = optimize(apply_rules(preprocessResult(
                                       h[i], preprocessAux())));
            if (!isNumberLike(h[i]))
                err("Array size must be fixed value", m);
            // Create a list of the coefficient associated with each
            // array index
            coefficients.push_back(decimalMul(coefficients.back(), h[i].val));
        }
    }
    // Tuples
    else {
        int startc;
        // Handle the (fun <fun_astnode> args...) case
        if (node.val == "fun") {
            startc = 1;
            h = listfyStorageAccess(node.args[0]);
        }
        // Handle the (<fun_name> args...) case, which
        // the serpent parser produces when the function
        // is a simple name and not a complex astnode
        else {
            startc = 0;
            h = listfyStorageAccess(token(node.val, m));
        }
        svObj sub = pre;
        sub.globalOffset = "0";
        // Evaluate tuple elements recursively
        for (unsigned i = startc; i < node.args.size(); i++) {
            sub = getStorageVars(sub,
                                 node.args[i],
                                 prefix+h[0].val.substr(2)+".",
                                 i-1);
        }
        coefficients.push_back(sub.globalOffset);
        for (unsigned i = h.size() - 1; i >= 1; i--) {
            // Array sizes must be constant or at least arithmetically
            // evaluable at compile time
            h[i] = optimize(apply_rules(preprocessResult(
                                      h[i], preprocessAux())));
            if (!isNumberLike(h[i]))
               err("Array size must be fixed value", m);
            // Create a list of the coefficient associated with each
            // array index
            coefficients.push_back(decimalMul(coefficients.back(), h[i].val));
        }
        pre.offsets = sub.offsets;
        pre.coefficients = sub.coefficients;
        pre.nonfinal = sub.nonfinal;
        pre.nonfinal[prefix+h[0].val.substr(2)] = true;
    }
    pre.coefficients[prefix+h[0].val.substr(2)] = coefficients;
    pre.offsets[prefix+h[0].val.substr(2)] = pre.globalOffset;
    pre.indices[prefix+h[0].val.substr(2)] = index;
    if (decimalGt(tt176, coefficients.back()))
        pre.globalOffset = decimalAdd(pre.globalOffset, coefficients.back());
    return pre;
}

// Transform a node of the form (call to funid vars...) into
// a call

#define psn std::pair<std::string, Node>

Node call_transform(Node node, std::string op) {
    Metadata m = node.metadata;
    // We're gonna make lots of temporary variables,
    // so set up a unique flag for them
    std::string prefix = "_temp"+mkUniqueToken()+"_";
    // kwargs = map of special arguments
    std::map<std::string, Node> kwargs;
    kwargs["value"] = token("0", m);
    kwargs["gas"] = parseLLL("(- (gas) 25)");
    std::vector<Node> args;
    for (unsigned i = 0; i < node.args.size(); i++) {
        if (node.args[i].val == "=" || node.args[i].val == "set") {
            if (node.args[i].args.size() != 2)
                err("Malformed set", m);
            kwargs[node.args[i].args[0].val] = node.args[i].args[1];
        }
        else args.push_back(node.args[i]);
    }
    if (args.size() < 2) err("Too few arguments for call!", m);
    kwargs["to"] = args[0];
    kwargs["funid"] = args[1];
    std::vector<Node> inputs;
    for (unsigned i = 2; i < args.size(); i++) {
        inputs.push_back(args[i]);
    }
    std::vector<psn> with;
    std::vector<Node> precompute;
    std::vector<Node> post;
    if (kwargs.count("data")) {
        if (!kwargs.count("datasz")) err("Required param datasz", m);
        // The strategy here is, we store the function ID byte at the index
        // before the start of the byte, but then we store the value that was
        // there before and reinstate it once the process is over
        // store data: data array start
        with.push_back(psn(prefix+"data", kwargs["data"]));
        // store data: prior: data array - 32
        Node prior = astnode("sub", token(prefix+"data", m), token("32", m), m);
        with.push_back(psn(prefix+"prior", prior));
        // store data: priormem: data array - 32 prior memory value
        Node priormem = astnode("mload", token(prefix+"prior", m), m);
        with.push_back(psn(prefix+"priormem", priormem));
        // post: reinstate prior mem at data array - 32
        post.push_back(astnode("mstore", 
                               token(prefix+"prior", m),
                               token(prefix+"priormem", m),
                               m));
        // store data: datastart: data array - 1
        Node datastart = astnode("sub",
                                 token(prefix+"data", m),
                                 token("1", m),
                                 m);
        with.push_back(psn(prefix+"datastart", datastart));
        // push funid byte to datastart
        precompute.push_back(astnode("mstore8", 
                                     token(prefix+"datastart", m),
                                     kwargs["funid"],
                                     m));
        // set data array start loc
        kwargs["datain"] = token(prefix+"datastart", m);
        kwargs["datainsz"] = astnode("add", 
                                     token("1", m),
                                     astnode("mul",
                                             token("32", m),
                                             kwargs["datasz"],
                                             m),
                                     m);
    }
    else {
        // Here, there is no data array, instead there are function arguments.
        // This actually lets us be much more efficient with how we set things
        // up.
        // Pre-declare variables; relies on declared variables being sequential
        precompute.push_back(astnode("declare",
                                     token(prefix+"prebyte", m),
                                     m));
        for (unsigned i = 0; i < inputs.size(); i++) {
            precompute.push_back(astnode("declare",
                                         token(prefix+unsignedToDecimal(i), m),
                                         m));
        }
        // Set up variables to store the function arguments, and store the
        // function ID at the byte before the start
        Node datastart = astnode("add",
                                 token("31", m),
                                 astnode("ref",
                                         token(prefix+"prebyte", m),
                                         m),
                                 m);
        precompute.push_back(astnode("mstore8",
                                     datastart,
                                     kwargs["funid"],
                                     m));
        for (unsigned i = 0; i < inputs.size(); i++) {
            precompute.push_back(astnode("set", 
                                         token(prefix+unsignedToDecimal(i), m),
                                         inputs[i],
                                         m));

        }
        kwargs["datain"] = datastart;
        kwargs["datainsz"] = token(unsignedToDecimal(inputs.size()*32+1), m);
    }
    if (!kwargs.count("outsz")) {
        kwargs["dataout"] = astnode("ref", token(prefix+"dataout", m), m);
        kwargs["dataoutsz"] = token("32", node.metadata);
        post.push_back(astnode("get", token(prefix+"dataout", m), m));
    }
    else {
        kwargs["dataout"] = kwargs["out"];
        kwargs["dataoutsz"] = kwargs["outsz"];
        post.push_back(astnode("ref", token(prefix+"dataout", m), m));
    }
    // Set up main call
    std::vector<Node> main;
    for (unsigned i = 0; i < precompute.size(); i++) {
        main.push_back(precompute[i]);
    }
    std::vector<Node> call;
    call.push_back(kwargs["gas"]);
    call.push_back(kwargs["to"]);
    call.push_back(kwargs["value"]);
    call.push_back(kwargs["datain"]);
    call.push_back(kwargs["datainsz"]);
    call.push_back(kwargs["dataout"]);
    call.push_back(kwargs["dataoutsz"]);
    main.push_back(astnode("pop", astnode("~"+op, call, m), m));
    for (unsigned i = 0; i < post.size(); i++) {
        main.push_back(post[i]);
    }
    Node mainNode = astnode("seq", main, node.metadata);
    // Add with variables
    for (int i = with.size() - 1; i >= 0; i--) {
        mainNode = astnode("with",
                           token(with[i].first, m),
                           with[i].second,
                           mainNode,
                           m);
    }
    return mainNode;
}

// Preprocess input containing functions
//
// localExterns is a map of the form, eg,
//
// { x: { foo: 0, bar: 1, baz: 2 }, y: { qux: 0, foo: 1 } ... }
//
// Signifying that x.foo = 0, x.baz = 2, y.foo = 1, etc
//
// globalExterns is a one-level map, eg from above
//
// { foo: 1, bar: 1, baz: 2, qux: 0 }
//
// Note that globalExterns may be ambiguous
preprocessResult preprocess(Node inp) {
    inp = inp.args[0];
    Metadata m = inp.metadata;
    if (inp.val != "seq") {
        std::vector<Node> args;
        args.push_back(inp);
        inp = astnode("seq", args, m);
    }
    std::vector<Node> empty;
    Node init = astnode("seq", empty, m);
    Node shared = astnode("seq", empty, m);
    std::vector<Node> any;
    std::vector<Node> functions;
    preprocessAux out = preprocessAux();
    out.localExterns["self"] = std::map<std::string, int>();
    int functionCount = 0;
    int storageDataCount = 0;
    for (unsigned i = 0; i < inp.args.size(); i++) {
        Node obj = inp.args[i];
        // Functions
        if (obj.val == "def") {
            if (obj.args.size() == 0)
                err("Empty def", m);
            std::string funName = obj.args[0].val;
            // Init, shared and any are special functions
            if (funName == "init" || funName == "shared" || funName == "any") {
                if (obj.args[0].args.size())
                    err(funName+" cannot have arguments", m);
            }
            if (funName == "init") init = obj.args[1];
            else if (funName == "shared") shared = obj.args[1];
            else if (funName == "any") any.push_back(obj.args[1]);
            else {
                // Other functions
                functions.push_back(obj);
                out.localExterns["self"][obj.args[0].val] = functionCount;
                functionCount++;
            }
        }
        // Extern declarations
        else if (obj.val == "extern") {
            std::string externName = obj.args[0].args[0].val;
            Node al = obj.args[0].args[1];
            if (!out.localExterns.count(externName))
                out.localExterns[externName] = std::map<std::string, int>();
            for (unsigned i = 0; i < al.args.size(); i++) {
                out.globalExterns[al.args[i].val] = i;
                out.localExterns[externName][al.args[i].val] = i;
            }
        }
        // Storage variables/structures
        else if (obj.val == "data") {
            out.storageVars = getStorageVars(out.storageVars,
                                             obj.args[0],
                                             "",
                                             storageDataCount);
            storageDataCount += 1;
        }
        else any.push_back(obj);
    }
    std::vector<Node> main;
    if (shared.args.size()) main.push_back(shared);
    if (init.args.size()) main.push_back(init);

    std::vector<Node> code;
    if (shared.args.size()) code.push_back(shared);
    for (unsigned i = 0; i < any.size(); i++)
        code.push_back(any[i]);
    for (unsigned i = 0; i < functions.size(); i++)
        code.push_back(functions[i]);
    main.push_back(astnode("~return",
                           token("0", m),
                           astnode("lll",
                                   astnode("seq", code, m),
                                   token("0", m),
                                   m),
                           m));



    return preprocessResult(astnode("seq", main, inp.metadata), out);
}

// Transform "<variable>.<fun>(args...)" into
// (call <variable> <funid> args...)
Node dotTransform(Node node, preprocessAux aux) {
    Metadata m = node.metadata;
    Node pre = node.args[0].args[0];
    std::string post = node.args[0].args[1].val;
    if (node.args[0].args[1].type == ASTNODE)
        err("Function name must be static", m);
    // Search for as=? and call=code keywords
    std::string as = "";
    bool call_code = false;
    for (unsigned i = 1; i < node.args.size(); i++) {
        Node arg = node.args[i];
        if (arg.val == "=" || arg.val == "set") {
            if (arg.args[0].val == "as")
                as = arg.args[1].val;
            if (arg.args[0].val == "call" && arg.args[1].val == "code")
                call_code = true;
        }
    }
    if (pre.val == "self") {
        if (as.size()) err("Cannot use \"as\" when calling self!", m);
        as = pre.val;
    }
    std::vector<Node> args;
    args.push_back(pre);
    // Determine the funId assuming the "as" keyword was used
    if (as.size() > 0 && aux.localExterns.count(as)) {
        if (!aux.localExterns[as].count(post))
            err("Invalid call: "+printSimple(pre)+"."+post, m);
        std::string funid = unsignedToDecimal(aux.localExterns[as][post]);
        args.push_back(token(funid, m));
    }
    // Determine the funId otherwise
    else if (!as.size()) {
        if (!aux.globalExterns.count(post))
            err("Invalid call: "+printSimple(pre)+"."+post, m);
        std::string key = unsignedToDecimal(aux.globalExterns[post]);
        args.push_back(token(key, m));
    }
    else err("Invalid call: "+printSimple(pre)+"."+post, m);
    for (unsigned i = 1; i < node.args.size(); i++)
        args.push_back(node.args[i]);
    return astnode(call_code ? "call_code" : "call", args, m);
}

// Transform an access of the form self.bob, self.users[5], etc into
// a storage access
//
// There exist two types of objects: finite objects, and infinite
// objects. Finite objects are packed optimally tightly into storage
// accesses; for example:
//
// data obj[100](a, b[2][4], c)
//
// obj[0].a -> 0
// obj[0].b[0][0] -> 1
// obj[0].b[1][3] -> 8
// obj[45].c -> 459
//
// Infinite objects are accessed by sha3([v1, v2, v3 ... ]), where
// the values are a list of array indices and keyword indices, for
// example:
// data obj[](a, b[2][4], c)
// data obj2[](a, b[][], c)
//
// obj[0].a -> sha3([0, 0, 0])
// obj[5].b[1][3] -> sha3([0, 5, 1, 1, 3])
// obj[45].c -> sha3([0, 45, 2])
// obj2[0].a -> sha3([1, 0, 0])
// obj2[5].b[1][3] -> sha3([1, 5, 1, 1, 3])
// obj2[45].c -> sha3([1, 45, 2])
Node storageTransform(Node node, preprocessAux aux, bool mapstyle=false) {
    Metadata m = node.metadata;
    // Get a list of all of the "access parameters" used in order
    // eg. self.users[5].cow[4][m[2]][woof] -> 
    //         [--self, --users, 5, --cow, 4, m[2], woof]
    std::vector<Node> hlist = listfyStorageAccess(node);
    // For infinite arrays, the terms array will just provide a list
    // of indices. For finite arrays, it's a list of index*coefficient
    std::vector<Node> terms;
    std::string offset = "0";
    std::string prefix = "";
    std::string varPrefix = "_temp"+mkUniqueToken()+"_";
    int c = 0;
    std::vector<std::string> coefficients;
    coefficients.push_back("");
    for (unsigned i = 1; i < hlist.size(); i++) {
        // We pre-add the -- flag to parameter-like terms. For example,
        // self.users[m] -> [--self, --users, m]
        // self.users.m -> [--self, --users, --m]
        if (hlist[i].val.substr(0, 2) == "--") {
            prefix += hlist[i].val.substr(2) + ".";
            std::string tempPrefix = prefix.substr(0, prefix.size()-1);
            if (!aux.storageVars.offsets.count(tempPrefix))
                return node;
            if (c < (signed)coefficients.size() - 1)
                err("Too few array index lookups", m);
            if (c > (signed)coefficients.size() - 1)
                err("Too many array index lookups", m);
            coefficients = aux.storageVars.coefficients[tempPrefix];
            // If the size of an object exceeds 2^176, we make it an infinite
            // array
            if (decimalGt(coefficients.back(), tt176) && !mapstyle)
                return storageTransform(node, aux, true);
            offset = decimalAdd(offset, aux.storageVars.offsets[tempPrefix]);
            c = 0;
            if (mapstyle)
                terms.push_back(token(unsignedToDecimal(
                    aux.storageVars.indices[tempPrefix])));
        }
        else if (mapstyle) {
            terms.push_back(hlist[i]);
            c += 1;
        }
        else {
            if (c > (signed)coefficients.size() - 2)
                err("Too many array index lookups", m);
            terms.push_back(
                astnode("mul", 
                        hlist[i],
                        token(coefficients[coefficients.size() - 2 - c], m),
                        m));
                                    
            c += 1;
        }
    }
    if (aux.storageVars.nonfinal.count(prefix.substr(0, prefix.size()-1)))
        err("Storage variable access not deep enough", m);
    if (c < (signed)coefficients.size() - 1) {
        err("Too few array index lookups", m);
    }
    if (c > (signed)coefficients.size() - 1) {
        err("Too many array index lookups", m);
    }
    if (mapstyle) {
        // We pre-declare variables, relying on the idea that sequentially
        // declared variables are doing to appear beside each other in
        // memory
        std::vector<Node> main;
        for (unsigned i = 0; i < terms.size(); i++)
            main.push_back(astnode("declare",
                                   token(varPrefix+unsignedToDecimal(i), m),
                                   m));
        for (unsigned i = 0; i < terms.size(); i++)
            main.push_back(astnode("set",
                                   token(varPrefix+unsignedToDecimal(i), m),
                                   terms[i],
                                   m));
        main.push_back(astnode("ref", token(varPrefix+"0", m), m));
        Node sz = token(unsignedToDecimal(terms.size()), m);
        return astnode("sload",
                       astnode("sha3",
                               astnode("seq", main, m),
                               sz,
                               m),
                       m);
    }
    else {
        // We add up all the index*coefficients
        Node out = token(offset, node.metadata);
        for (unsigned i = 0; i < terms.size(); i++) {
            std::vector<Node> temp;
            temp.push_back(out);
            temp.push_back(terms[i]);
            out = astnode("add", temp, node.metadata);
        }
        std::vector<Node> temp2;
        temp2.push_back(out);
        return astnode("sload", temp2, node.metadata);
    }
}


// Recursively applies rewrite rules
Node apply_rules(preprocessResult pr) {
    Node node = pr.first;
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
    // Assignment transformations
    for (int i = 0; i < 9999; i++) {
        if (setters[i][0] == "---END---") break;
        if (node.val == setters[i][0]) {
            node = astnode("=",
                           node.args[0],
                           astnode(setters[i][1],
                                   node.args[0],
                                   node.args[1],
                                   node.metadata),
                           node.metadata);
        }
    }
    // Special storage transformation
    if (isNodeStorageVariable(node)) {
        node = storageTransform(node, pr.second);
    }
    if (node.val == "=" && isNodeStorageVariable(node.args[0])) {
        Node t = storageTransform(node.args[0], pr.second);
        if (t.val == "sload") {
            std::vector<Node> o;
            o.push_back(t.args[0]);
            o.push_back(node.args[1]);
            node = astnode("sstore", o, node.metadata);
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
            pos = 0;
        }
    }
    // Special transformations
    if (node.val == "outer") {
        pr = preprocess(node);
        node = pr.first;
    }
    if (node.val == "array_lit")
        node = array_lit_transform(node);
    if (node.val == "fun" && node.args[0].val == ".") {
        node = dotTransform(node, pr.second);
    }
    if (node.val == "call")
        node = call_transform(node, "call");
    if (node.val == "call_code")
        node = call_transform(node, "call_code");
    if (node.type == ASTNODE) {
		unsigned i = 0;
        if (node.val == "set" || node.val == "ref" 
                || node.val == "get" || node.val == "with"
                || node.val == "def" || node.val == "declare") {
            node.args[0].val = "'" + node.args[0].val;
            i = 1;
        }
        if (node.val == "def") {
            for (unsigned j = 0; j < node.args[0].args.size(); j++) {
                if (node.args[0].args[j].val == ":") {
                    node.args[0].args[j].val = "kv";
                    node.args[0].args[j].args[0].val =
                         "'" + node.args[0].args[j].args[0].val;
                }
                else {
                    node.args[0].args[j].val = "'" + node.args[0].args[j].val;
                }
            }
        }
        for (; i < node.args.size(); i++) {
            node.args[i] =
                apply_rules(preprocessResult(node.args[i], pr.second));
        }
    }
    else if (node.type == TOKEN && !isNumberLike(node)) {
        node.val = "'" + node.val;
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

// Compile-time arithmetic calculations
Node optimize(Node inp) {
    if (inp.type == TOKEN) {
        Node o = tryNumberize(inp);
        if (decimalGt(o.val, tt256, true))
            err("Value too large (exceeds 32 bytes or 2^256)", inp.metadata);
        return o;
    }
	for (unsigned i = 0; i < inp.args.size(); i++) {
        inp.args[i] = optimize(inp.args[i]);
    }
    // Degenerate cases for add and mul
    if (inp.args.size() == 2) {
        if (inp.val == "add" && inp.args[0].type == TOKEN && 
                inp.args[0].val == "0") {
            inp = inp.args[1];
        }
        if (inp.val == "add" && inp.args[1].type == TOKEN && 
                inp.args[1].val == "0") {
            inp = inp.args[0];
        }
        if (inp.val == "mul" && inp.args[0].type == TOKEN && 
                inp.args[0].val == "1") {
            inp = inp.args[1];
        }
        if (inp.val == "mul" && inp.args[1].type == TOKEN && 
                inp.args[1].val == "1") {
            inp = inp.args[0];
        }
    }
    // Arithmetic computation
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
      else if (inp.val == "exp") {
          o = decimalModExp(inp.args[0].val, inp.args[1].val, tt256);
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
                std::string sz = unsignedToDecimal(inp.args.size());
                if (decimalGt(valid[i][1], sz)) {
                    err("Too few arguments for "+inp.val, inp.metadata);   
                }
                if (decimalGt(sz, valid[i][2])) {
                    err("Too many arguments for "+inp.val, inp.metadata);   
                }
            }
            i++;
        }
    }
	for (unsigned i = 0; i < inp.args.size(); i++) validate(inp.args[i]);
    return inp;
}

Node postValidate(Node inp) {
    if (inp.type == ASTNODE) {
        if (inp.val == ".")
            err("Invalid object member (ie. a foo.bar not mapped to anything)",
                inp.metadata);
        for (unsigned i = 0; i < inp.args.size(); i++)
            postValidate(inp.args[i]);
    }
    return inp;
}

Node outerWrap(Node inp) {
    std::vector<Node> args;
    args.push_back(inp);
    return astnode("outer", args, inp.metadata);
}

Node rewrite(Node inp) {
    return postValidate(optimize(apply_rules(preprocessResult(
                validate(outerWrap(inp)), preprocessAux()))));
}

Node rewriteChunk(Node inp) {
    return postValidate(optimize(apply_rules(preprocessResult(
                validate(inp), preprocessAux()))));
}

using namespace std;
