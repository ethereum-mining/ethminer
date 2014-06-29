#include <stdio.h>
#include <iostream>
#include <vector>
#include <map>
#include "util.h"
#include "bignum.h"
#include "opcodes.h"

struct programAux {
    std::map<std::string, std::string> vars;
    bool allocUsed;
    bool calldataUsed;
    int step;
    int labelLength;
};

struct programData {
    programAux aux;
    Node code;
};

programAux Aux() {
    programAux o;
    o.allocUsed = false;
    o.calldataUsed = false;
    o.step = 0;
    return o;
}

programData pd(programAux aux = Aux(), Node code=token("_")) {
    programData o;
    o.aux = aux;
    o.code = code;
    return o;
}

Node multiToken(Node nodes[], int len, Metadata met) {
    std::vector<Node> out;
    for (int i = 0; i < len; i++) {
        out.push_back(nodes[i]);
    }
    return astnode("_", out, met);
}

Node finalize(programData c);

// Turns LLL tree into tree of code fragments
programData opcodeify(Node node, programAux aux=Aux()) {
    std::string symb = "_"+mkUniqueToken();
    Metadata m = node.metadata;
    // Numbers
    if (node.type == TOKEN) {
        return pd(aux, nodeToNumeric(node));
    }
    else if (node.val == "ref" || node.val == "get" || node.val == "set") {
        std::string varname = node.args[0].val;
        if (!aux.vars.count(varname)) {
            aux.vars[varname] = intToDecimal(aux.vars.size() * 32);
        }
        if (varname == "msg.data") aux.calldataUsed = true;
        // Set variable
        if (node.val == "set") {
             programData sub = opcodeify(node.args[1], aux);
             Node nodelist[] = {
                 sub.code,
                 token(aux.vars[varname], m),
                 token("MSTORE", m),
             };
             return pd(sub.aux, multiToken(nodelist, 3, m));                   
        }
        // Get variable
        else if (node.val == "get") {
             Node nodelist[] = 
                  { token(aux.vars[varname], m), token("MLOAD", m) };
             return pd(aux, multiToken(nodelist, 2, m));
        }
        // Refer variable
        else return pd(aux, token(aux.vars[varname], m));
    }
    // Code blocks
    if (node.val == "lll" && node.args.size() == 2) {
        if (node.args[1].val != "0") aux.allocUsed = true;
        std::vector<Node> o;
        o.push_back(finalize(opcodeify(node.args[0])));
        programData sub = opcodeify(node.args[1], aux);
        Node code = astnode("____CODE", o, m);
        Node nodelist[] = {
            token("$begincode"+symb+".endcode"+symb, m), token("DUP", m),
            sub.code,
            token("$begincode"+symb, m), token("CODECOPY", m),
            token("$endcode"+symb, m), token("JUMP", m),
            token("~begincode"+symb, m), code, token("~endcode"+symb, m)
        };
        return pd(sub.aux, multiToken(nodelist, 10, m));
    }
    std::vector<Node> subs;
	for (unsigned i = 0; i < node.args.size(); i++) {
        programData sub = opcodeify(node.args[i], aux);
        aux = sub.aux;
        subs.push_back(sub.code);
    }
    // Debug
    if (node.val == "debug") {
        Node nodelist[] = {
            subs[0],
            token("DUP", m), token("POP", m), token("POP", m)
        };
        return pd(aux, multiToken(nodelist, 4, m));
    }
    // Seq of multiple statements
    if (node.val == "seq") {
        return pd(aux, astnode("_", subs, m));
    }
    // 2-part conditional (if gets rewritten to unless in rewrites)
    else if (node.val == "unless" && node.args.size() == 2) {
        Node nodelist[] = {
            subs[0],
            token("$endif"+symb, m), token("JUMPI", m),
            subs[1],
            token("~endif"+symb, m)
        };
        return pd(aux, multiToken(nodelist, 5, m));
    }
    // 3-part conditional
    else if (node.val == "if" && node.args.size() == 3) {
        Node nodelist[] = {
            subs[0],
            token("NOT", m), token("$else"+symb, m), token("JUMPI", m),
            subs[1],
            token("$endif"+symb, m), token("JUMP", m), token("~else"+symb, m),
            subs[2],
            token("~endif"+symb, m)
        };
        return pd(aux, multiToken(nodelist, 10, m));
    }
    // While (rewritten to this in rewrites)
    else if (node.val == "until") {
        Node nodelist[] = {
            token("~beg"+symb, m),
            subs[0],
            token("$end"+symb, m), token("JUMPI", m),
            subs[1],
            token("$beg"+symb, m), token("JUMP", m), token("~end"+symb, m)
        };
        return pd(aux, multiToken(nodelist, 8, m));
    }
    // Memory allocations
    else if (node.val == "alloc") {
        aux.allocUsed = true;
        Node nodelist[] = {
            subs[0],
            token("MSIZE", m), token("SWAP", m), token("MSIZE", m),
            token("ADD", m), 
            token("0", m), token("SWAP", m), token("MSTORE", m)
        };
        return pd(aux, multiToken(nodelist, 8, m));
    }
    // Array literals
    else if (node.val == "array_lit") {
        aux.allocUsed = true;
        std::vector<Node> nodes;
        if (!subs.size()) {
            nodes.push_back(token("MSIZE", m));
            return pd(aux, astnode("_", nodes, m));
        }
        nodes.push_back(token("MSIZE", m));
        nodes.push_back(token("0", m));
        nodes.push_back(token("MSIZE", m));
        nodes.push_back(token(intToDecimal(subs.size() * 32 - 1), m));
        nodes.push_back(token("ADD", m));
        nodes.push_back(token("MSTORE8", m));
		for (unsigned i = 0; i < subs.size(); i++) {
            nodes.push_back(token("DUP", m));
            nodes.push_back(subs[i]);
            nodes.push_back(token("SWAP", m));
            if (i > 0) {
                nodes.push_back(token(intToDecimal(i * 32), m));
                nodes.push_back(token("ADD", m));
            }
            nodes.push_back(token("MSTORE", m));
        }
        return pd(aux, astnode("_", nodes, m));
    }
    // All other functions/operators
    else {
        std::vector<Node> subs2;
        while (subs.size()) {
            subs2.push_back(subs.back());
            subs.pop_back();
        }
        subs2.push_back(token(upperCase(node.val), m));
        return pd(aux, astnode("_", subs2, m));
    }
}

// Adds necessary wrappers to a program
Node finalize(programData c) {
    std::vector<Node> bottom;
    Metadata m = c.code.metadata;
    // If we are using both alloc and variables, we need to pre-zfill
    // some memory
    if (c.aux.allocUsed && c.aux.vars.size() > 0) {
        Node nodelist[] = {
            token("0", m), 
            token(intToDecimal(c.aux.vars.size() * 32 - 1)),
            token("MSTORE8", m)
        };
        bottom.push_back(multiToken(nodelist, 3, m));
    }
    // If msg.data is being used as an array, then we need to copy it
    if (c.aux.calldataUsed) {
        Node nodelist[] = {
            token("MSIZE", m), token("CALLDATASIZE", m), token("MSIZE", m),
            token("0", m), token("CALLDATACOPY", m),
            token(c.aux.vars["msg.data"], m), token("MSTORE", m)
        };
        bottom.push_back(multiToken(nodelist, 7, m));
    }
    // The actual code
    bottom.push_back(c.code);
    return astnode("_", bottom, m);
}

//LLL -> code fragment tree
Node buildFragmentTree(Node node) {
    return finalize(opcodeify(node));
}


// Builds a dictionary mapping labels to variable names
programAux buildDict(Node program, programAux aux, int labelLength) {
    Metadata m = program.metadata;
    // Token
    if (program.type == TOKEN) {
        if (isNumberLike(program)) {
            aux.step += 1 + toByteArr(program.val, m).size();
        }
        else if (program.val[0] == '~') {
            aux.vars[program.val.substr(1)] = intToDecimal(aux.step);
        }
        else if (program.val[0] == '$') {
            aux.step += labelLength + 1;
        }
        else aux.step += 1;
    }
    // A sub-program (ie. LLL)
    else if (program.val == "____CODE") {
        programAux auks = Aux();
		for (unsigned i = 0; i < program.args.size(); i++) {
            auks = buildDict(program.args[i], auks, labelLength);
        }
        for (std::map<std::string,std::string>::iterator it=auks.vars.begin();
             it != auks.vars.end();
             it++) {
            aux.vars[(*it).first] = (*it).second;
        }
        aux.step += auks.step;
    }
    // Normal sub-block
    else {
		for (unsigned i = 0; i < program.args.size(); i++) {
            aux = buildDict(program.args[i], aux, labelLength);
        }
    }
    return aux;
}

// Applies that dictionary
Node substDict(Node program, programAux aux, int labelLength) {
    Metadata m = program.metadata;
    std::vector<Node> out;
    std::vector<Node> inner;
    if (program.type == TOKEN) {
        if (program.val[0] == '$') {
            std::string tokStr = "PUSH"+intToDecimal(labelLength);
            out.push_back(token(tokStr, m));
            int dotLoc = program.val.find('.');
            if (dotLoc == -1) {
                std::string val = aux.vars[program.val.substr(1)];
                inner = toByteArr(val, m, labelLength);
            }
            else {
                std::string start = aux.vars[program.val.substr(1, dotLoc-1)],
                            end = aux.vars[program.val.substr(dotLoc + 1)],
                            dist = decimalSub(end, start);
                inner = toByteArr(dist, m, labelLength);
            }
            out.push_back(astnode("_", inner, m));
        }
        else if (program.val[0] == '~') { }
        else if (isNumberLike(program)) {
            inner = toByteArr(program.val, m);
            out.push_back(token("PUSH"+intToDecimal(inner.size())));
            out.push_back(astnode("_", inner, m));
        }
        else return program;
    }
    else {
		for (unsigned i = 0; i < program.args.size(); i++) {
            Node n = substDict(program.args[i], aux, labelLength);
            if (n.type == TOKEN || n.args.size()) out.push_back(n);
        }
    }
    return astnode("_", out, m);
}

// Compiled fragtree -> compiled fragtree without labels
Node dereference(Node program) {
    int sz = treeSize(program) * 4;
    int labelLength = 1;
    while (sz >= 256) { labelLength += 1; sz /= 256; }
    programAux aux = buildDict(program, Aux(), labelLength);
    return substDict(program, aux, labelLength);
}

// Dereferenced fragtree -> opcodes
std::vector<Node> flatten(Node derefed) {
    std::vector<Node> o;
    if (derefed.type == TOKEN) {
        o.push_back(derefed);
    }
    else {
		for (unsigned i = 0; i < derefed.args.size(); i++) {
            std::vector<Node> oprime = flatten(derefed.args[i]);
			for (unsigned j = 0; j < oprime.size(); j++) o.push_back(oprime[j]);
        }
    }
    return o;
}

// Opcodes -> bin
std::string serialize(std::vector<Node> codons) {
    std::string o;
	for (unsigned i = 0; i < codons.size(); i++) {
        int v;
        if (isNumberLike(codons[i])) {
            v = decimalToInt(codons[i].val);
        }
        else if (codons[i].val.substr(0,4) == "PUSH") {
            v = 95 + decimalToInt(codons[i].val.substr(4));
        }
        else {
            v = opcode(codons[i].val);
        }
        o += (char)v;
    }
    return o;
}

// Bin -> opcodes
std::vector<Node> deserialize(std::string ser) {
    std::vector<Node> o;
    int backCount = 0;
	for (unsigned i = 0; i < ser.length(); i++) {
        unsigned char v = (unsigned char)ser[i];
        std::string oper = op((int)v);
        if (oper != "" && backCount <= 0) o.push_back(token(oper));
        else if (v >= 96 && v < 128 && backCount <= 0) {
            o.push_back(token("PUSH"+intToDecimal(v - 95)));
        }
        else o.push_back(token(intToDecimal(v)));
        if (v >= 96 && v < 128 && backCount <= 0) {
            backCount = v - 95;
        }
        else backCount--;
    }
    return o;
}

// Fragtree -> bin
std::string assemble(Node fragTree) {
    return serialize(flatten(dereference(fragTree)));
}

// Fragtree -> tokens
std::vector<Node> prettyAssemble(Node fragTree) {
    return flatten(dereference(fragTree));
}

// LLL -> bin
std::string compileLLL(Node program) {
    return assemble(buildFragmentTree(program));
}

// LLL -> tokens
std::vector<Node> prettyCompileLLL(Node program) {
    return prettyAssemble(buildFragmentTree(program));
}

// Converts a list of integer values to binary transaction data
std::string encodeDatalist(std::vector<std::string> vals) {
    std::string o;
	for (unsigned i = 0; i < vals.size(); i++) {
        std::vector<Node> n = toByteArr(strToNumeric(vals[i]), Metadata(), 32);
		for (unsigned j = 0; j < n.size(); j++) {
            int v = decimalToInt(n[j].val);
            o += (char)v;
        }
    }
    return o;
}

// Converts binary transaction data into a list of integer values
std::vector<std::string> decodeDatalist(std::string ser) {
    std::vector<std::string> out;
	for (unsigned i = 0; i < ser.length(); i+= 32) {
        std::string o = "0";
		for (unsigned j = i; j < i + 32; j++) {
            int vj = (int)(unsigned char)ser[j];
            o = decimalAdd(decimalMul(o, "256"), intToDecimal(vj));
        }
        out.push_back(o);
    }
    return out;
}
