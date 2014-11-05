#include <stdio.h>
#include <iostream>
#include <vector>
#include <map>
#include "util.h"
#include "bignum.h"
#include "opcodes.h"

struct programAux {
    std::map<std::string, std::string> vars;
    int nextVarMem;
    bool allocUsed;
    bool calldataUsed;
    int step;
    int labelLength;
    int functionCount;
};

struct programVerticalAux {
    int height;
    std::map<std::string, int> dupvars;
    std::map<std::string, int> funvars;
};

struct programData {
    programAux aux;
    Node code;
    int outs;
};

programAux Aux() {
    programAux o;
    o.allocUsed = false;
    o.calldataUsed = false;
    o.step = 0;
    o.nextVarMem = 32;
    o.functionCount = 0;
    return o;
}

programVerticalAux verticalAux() {
    programVerticalAux o;
    o.height = 0;
    o.dupvars = std::map<std::string, int>();
    o.funvars = std::map<std::string, int>();
    return o;
}

programData pd(programAux aux = Aux(), Node code=token("_"), int outs=0) {
    programData o;
    o.aux = aux;
    o.code = code;
    o.outs = outs;
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

Node popwrap(Node node) {
    Node nodelist[] = {
        node,
        token("POP", node.metadata)
    };
    return multiToken(nodelist, 2, node.metadata);
}

// Turns LLL tree into tree of code fragments
programData opcodeify(Node node,
                      programAux aux=Aux(),
                      programVerticalAux vaux=verticalAux()) {
    std::string symb = "_"+mkUniqueToken();
    Metadata m = node.metadata;
    // Numbers
    if (node.type == TOKEN) {
        return pd(aux, nodeToNumeric(node), 1);
    }
    else if (node.val == "ref" || node.val == "get" ||
             node.val == "set" || node.val == "declare") {
        std::string varname = node.args[0].val;
        if (!aux.vars.count(varname)) {
            aux.vars[varname] = unsignedToDecimal(aux.nextVarMem);
            aux.nextVarMem += 32;
        }
        if (varname == "'msg.data") aux.calldataUsed = true;
        // Set variable
        if (node.val == "set") {
            programData sub = opcodeify(node.args[1], aux, vaux);
            if (!sub.outs)
                err("Value to set variable must have nonzero arity!", m);
            if (vaux.dupvars.count(node.args[0].val)) {
                int h = vaux.height - vaux.dupvars[node.args[0].val];
                if (h > 16) err("Too deep for stack variable (max 16)", m);
                Node nodelist[] = {
                    sub.code,
                    token("SWAP"+unsignedToDecimal(h), m),
                    token("POP", m)
                };
                return pd(sub.aux, multiToken(nodelist, 3, m), 0);                   
            }
            Node nodelist[] = {
                sub.code,
                token(sub.aux.vars[varname], m),
                token("MSTORE", m),
            };
            return pd(sub.aux, multiToken(nodelist, 3, m), 0);                   
        }
        // Get variable
        else if (node.val == "get") {
             if (vaux.dupvars.count(node.args[0].val)) {
                 int h = vaux.height - vaux.dupvars[node.args[0].val];
                if (h > 16) err("Too deep for stack variable (max 16)", m);
                return pd(aux, token("DUP"+unsignedToDecimal(h)), 1);                   
            }
            Node nodelist[] = 
                 { token(aux.vars[varname], m), token("MLOAD", m) };
            return pd(aux, multiToken(nodelist, 2, m), 1);
        }
        // Refer variable
        else if (node.val == "ref") {
            if (vaux.dupvars.count(node.args[0].val))
                err("Cannot ref stack variable!", m);
            return pd(aux, token(aux.vars[varname], m), 1);
        }
        // Declare variable
        else {
            Node nodelist[] = { };
            return pd(aux, multiToken(nodelist, 0, m), 0);
        }
    }
    // Define functions (TODO: eventually move to rewriter.cpp, keep
    // compiler pure LLL)
    if (node.val == "def") {
        std::vector<std::string> varNames;
        std::vector<int> varSizes;
        bool useLt32 = false;
        int totalSz = 0;
        if (node.args.size() != 2)
            err("Malformed def!", m);
        // Collect the list of variable names and variable byte counts
        for (unsigned i = 0; i < node.args[0].args.size(); i++) {
            if (node.args[0].args[i].val == "kv") {
                if (node.args[0].args[i].args.size() != 2)
                    err("Malformed def!", m);
                varNames.push_back(node.args[0].args[i].args[0].val);
                varSizes.push_back(
                    decimalToUnsigned(node.args[0].args[i].args[1].val));
                if (varSizes.back() > 32)
                    err("Max argument width: 32 bytes", m);
                useLt32 = true;
            }
            else {
                varNames.push_back(node.args[0].args[i].val);
                varSizes.push_back(32);
            }
            aux.vars[varNames.back()] = unsignedToDecimal(aux.nextVarMem + 32 * i);
            totalSz += varSizes.back();
        }
        int functionCount = aux.functionCount;
        int nextVarMem = aux.nextVarMem;
        aux.nextVarMem += 32 * varNames.size();
        aux.functionCount += 1;
        programData inner;
        // If we're only using 32-byte variables, then great, just copy
        // over the calldata!
        if (!useLt32) {
            programData sub = opcodeify(node.args[1], aux, vaux);
            Node nodelist[] = {
                token(unsignedToDecimal(totalSz), m),
                token("1", m),
                token(unsignedToDecimal(nextVarMem), m),
                token("CALLDATACOPY", m),
                sub.code
            };
            inner = pd(sub.aux, multiToken(nodelist, 5, m), 0);
        }
        else {
            std::vector<Node> innerList;
            int cum = 1;
            for (unsigned i = 0; i < varNames.size();) {
                // If we get a series of 32-byte values, we calldatacopy them
                if (varSizes[i] == 32) {
                    unsigned until = i+1;
                    while (until < varNames.size() && varSizes[until] == 32)
                        until += 1;
                    innerList.push_back(token(unsignedToDecimal((until - i) * 32), m));
                    innerList.push_back(token(unsignedToDecimal(cum), m));
                    innerList.push_back(token(unsignedToDecimal(nextVarMem + i * 32), m));
                    innerList.push_back(token("CALLDATACOPY", m));
                    cum += (until - i) * 32;
                    i = until;
                }
                // Otherwise, we do a clever trick to extract the value
                else {
                    innerList.push_back(token(unsignedToDecimal(32 - varSizes[i]), m));
                    innerList.push_back(token("256", m));
                    innerList.push_back(token("EXP", m));
                    innerList.push_back(token(unsignedToDecimal(cum), m));
                    innerList.push_back(token("CALLDATALOAD", m));
                    innerList.push_back(token("DIV", m));
                    innerList.push_back(token(unsignedToDecimal(nextVarMem + i * 32), m));
                    innerList.push_back(token("MSTORE", m));
                    cum += varSizes[i];
                    i += 1;
                }
            }
            // If caller == origin, then it's from a tx, so unpack, otherwise
            // plain copy
            programData sub = opcodeify(node.args[1], aux, vaux);
            Node ilnode = astnode("", innerList, m);
            Node nodelist[] = {
                token(unsignedToDecimal(32 * varNames.size()), m),
                token("1", m),
                token(unsignedToDecimal(nextVarMem), m),
                token("CALLDATACOPY", m),
                token("CALLER", m),
                token("ORIGIN", m),
                token("EQ", m),
                token("ISZERO", m),
                token("$maincode"+symb, m),
                token("JUMPI", m),
                ilnode,
                token("~maincode"+symb, m),
                token("JUMPDEST", m),
                sub.code
            };
            inner = pd(sub.aux, multiToken(nodelist, 14, m), 0);
        }
        // Check if the function call byte is the same
        Node nodelist2[] = {
            token("0", m),
            token("CALLDATALOAD", m),
            token("0", m),
            token("BYTE", m),
            token(unsignedToDecimal(functionCount), m),
            token("EQ", m),
            token("ISZERO", m),
            token("$endcode"+symb, m),
            token("JUMPI", m),
            inner.code,
            token("~endcode"+symb, m),
            token("JUMPDEST", m),
        };
        return pd(inner.aux, multiToken(nodelist2, 12, m), 0);
    }
    // Code blocks
    if (node.val == "lll" && node.args.size() == 2) {
        if (node.args[1].val != "0") aux.allocUsed = true;
        std::vector<Node> o;
        o.push_back(finalize(opcodeify(node.args[0])));
        programData sub = opcodeify(node.args[1], aux, vaux);
        Node code = astnode("____CODE", o, m);
        Node nodelist[] = {
            token("$begincode"+symb+".endcode"+symb, m), token("DUP1", m),
            token("$begincode"+symb, m), sub.code, token("CODECOPY", m),
            token("$endcode"+symb, m), token("JUMP", m),
            token("~begincode"+symb, m), code, 
            token("~endcode"+symb, m), token("JUMPDEST", m)
        };
        return pd(sub.aux, multiToken(nodelist, 11, m), 1);
    }
    // Stack variables
    if (node.val == "with") {
        programData initial = opcodeify(node.args[1], aux, vaux);
        programVerticalAux vaux2 = vaux;
        vaux2.dupvars[node.args[0].val] = vaux.height;
        vaux2.height += 1;
        if (!initial.outs)
            err("Initial variable value must have nonzero arity!", m);
        programData sub = opcodeify(node.args[2], initial.aux, vaux2);
        Node nodelist[] = {
            initial.code,
            sub.code
        };
        programData o = pd(sub.aux, multiToken(nodelist, 2, m), sub.outs);
        if (sub.outs)
            o.code.args.push_back(token("SWAP1", m));
        o.code.args.push_back(token("POP", m));
        return o;
    }
    // Seq of multiple statements
    if (node.val == "seq") {
        std::vector<Node> children;
        int lastOut = 0;
        for (unsigned i = 0; i < node.args.size(); i++) {
            programData sub = opcodeify(node.args[i], aux, vaux);
            aux = sub.aux;
            if (sub.outs == 1) {
                if (i < node.args.size() - 1) sub.code = popwrap(sub.code);
                else lastOut = 1;
            }
            children.push_back(sub.code);
        }
        return pd(aux, astnode("_", children, m), lastOut);
    }
    // 2-part conditional (if gets rewritten to unless in rewrites)
    else if (node.val == "unless" && node.args.size() == 2) {
        programData cond = opcodeify(node.args[0], aux, vaux);
        programData action = opcodeify(node.args[1], cond.aux, vaux);
        aux = action.aux;
        if (!cond.outs) err("Condition of if/unless statement has arity 0", m);
        if (action.outs) action.code = popwrap(action.code);
        Node nodelist[] = {
            cond.code,
            token("$endif"+symb, m), token("JUMPI", m),
            action.code,
            token("~endif"+symb, m), token("JUMPDEST", m)
        };
        return pd(aux, multiToken(nodelist, 6, m), 0);
    }
    // 3-part conditional
    else if (node.val == "if" && node.args.size() == 3) {
        programData ifd = opcodeify(node.args[0], aux, vaux);
        programData thend = opcodeify(node.args[1], ifd.aux, vaux);
        programData elsed = opcodeify(node.args[2], thend.aux, vaux);
        aux = elsed.aux;
        if (!ifd.outs)
            err("Condition of if/unless statement has arity 0", m);
        // Handle cases where one conditional outputs something
        // and the other does not
        int outs = (thend.outs && elsed.outs) ? 1 : 0;
        if (thend.outs > outs) thend.code = popwrap(thend.code);
        if (elsed.outs > outs) elsed.code = popwrap(elsed.code);
        Node nodelist[] = {
            ifd.code,
            token("ISZERO", m),
            token("$else"+symb, m), token("JUMPI", m),
            thend.code,
            token("$endif"+symb, m), token("JUMP", m),
            token("~else"+symb, m), token("JUMPDEST", m),
            elsed.code,
            token("~endif"+symb, m), token("JUMPDEST", m)
        };
        return pd(aux, multiToken(nodelist, 12, m), outs);
    }
    // While (rewritten to this in rewrites)
    else if (node.val == "until") {
        programData cond = opcodeify(node.args[0], aux, vaux);
        programData action = opcodeify(node.args[1], cond.aux, vaux);
        aux = action.aux;
        if (!cond.outs)
            err("Condition of while/until loop has arity 0", m);
        if (action.outs) action.code = popwrap(action.code);
        Node nodelist[] = {
            token("~beg"+symb, m), token("JUMPDEST", m),
            cond.code,
            token("$end"+symb, m), token("JUMPI", m),
            action.code,
            token("$beg"+symb, m), token("JUMP", m),
            token("~end"+symb, m), token("JUMPDEST", m),
        };
        return pd(aux, multiToken(nodelist, 10, m));
    }
    // Memory allocations
    else if (node.val == "alloc") {
        programData bytez = opcodeify(node.args[0], aux, vaux);
        aux = bytez.aux;
        if (!bytez.outs)
            err("Alloc input has arity 0", m);
        aux.allocUsed = true;
        Node nodelist[] = {
            bytez.code,
            token("MSIZE", m), token("SWAP1", m), token("MSIZE", m),
            token("ADD", m), 
            token("0", m), token("SWAP1", m), token("MSTORE", m)
        };
        return pd(aux, multiToken(nodelist, 8, m), 1);
    }
    // Array literals
    else if (node.val == "array_lit") {
        aux.allocUsed = true;
        std::vector<Node> nodes;
        if (!node.args.size()) {
            nodes.push_back(token("MSIZE", m));
            return pd(aux, astnode("_", nodes, m));
        }
        nodes.push_back(token("MSIZE", m));
        nodes.push_back(token("0", m));
        nodes.push_back(token("MSIZE", m));
        nodes.push_back(token(unsignedToDecimal(node.args.size() * 32 - 1), m));
        nodes.push_back(token("ADD", m));
        nodes.push_back(token("MSTORE8", m));
        for (unsigned i = 0; i < node.args.size(); i++) {
            Metadata m2 = node.args[i].metadata;
            nodes.push_back(token("DUP1", m2));
            programVerticalAux vaux2 = vaux;
            vaux2.height += 2;
            programData sub = opcodeify(node.args[i], aux, vaux2);
            if (!sub.outs)
                err("Array_lit item " + unsignedToDecimal(i) + " has zero arity", m2);
            aux = sub.aux;
            nodes.push_back(sub.code);
            nodes.push_back(token("SWAP1", m2));
            if (i > 0) {
                nodes.push_back(token(unsignedToDecimal(i * 32), m2));
                nodes.push_back(token("ADD", m2));
            }
            nodes.push_back(token("MSTORE", m2));
        }
        return pd(aux, astnode("_", nodes, m), 1);
    }
    // All other functions/operators
    else {
        std::vector<Node>  subs2;
        int depth = opinputs(upperCase(node.val));
        if (node.val != "debug") {
            if (depth == -1)
                err("Not a function or opcode: "+node.val, m);
            if ((int)node.args.size() != depth)
                err("Invalid arity for "+node.val, m);
        }
        for (int i = node.args.size() - 1; i >= 0; i--) {
            programVerticalAux vaux2 = vaux;
            vaux2.height = vaux.height - i - 1 + node.args.size();
            programData sub = opcodeify(node.args[i], aux, vaux2);
            aux = sub.aux;
            if (!sub.outs)
                err("Input "+unsignedToDecimal(i)+" has arity 0", sub.code.metadata);
            subs2.push_back(sub.code);
        }
        if (node.val == "debug") {
            subs2.push_back(token("DUP"+unsignedToDecimal(node.args.size()), m));
            for (int i = 0; i <= (int)node.args.size(); i++)
                subs2.push_back(token("POP", m));
        }
        else subs2.push_back(token(upperCase(node.val), m));
        int outdepth = node.val == "debug" ? 0 : opoutputs(upperCase(node.val));
        return pd(aux, astnode("_", subs2, m), outdepth);
    }
}

// Adds necessary wrappers to a program
Node finalize(programData c) {
    std::vector<Node> bottom;
    Metadata m = c.code.metadata;
    // If we are using both alloc and variables, we need to pre-zfill
    // some memory
    if ((c.aux.allocUsed || c.aux.calldataUsed) && c.aux.vars.size() > 0) {
        Node nodelist[] = {
            token("0", m), 
            token(unsignedToDecimal(c.aux.nextVarMem - 1)),
            token("MSTORE8", m)
        };
        bottom.push_back(multiToken(nodelist, 3, m));
    }
    // If msg.data is being used as an array, then we need to copy it
    if (c.aux.calldataUsed) {
        Node nodelist[] = {
            token("MSIZE", m), token("CALLDATASIZE", m), token("0", m),
            token("MSIZE", m), token("CALLDATACOPY", m),
            token(c.aux.vars["'msg.data"], m), token("MSTORE", m)
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
            aux.vars[program.val.substr(1)] = unsignedToDecimal(aux.step);
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
            std::string tokStr = "PUSH"+unsignedToDecimal(labelLength);
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
            out.push_back(token("PUSH"+unsignedToDecimal(inner.size())));
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
            v = decimalToUnsigned(codons[i].val);
        }
        else if (codons[i].val.substr(0,4) == "PUSH") {
            v = 95 + decimalToUnsigned(codons[i].val.substr(4));
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
            o.push_back(token("PUSH"+unsignedToDecimal(v - 95)));
        }
        else o.push_back(token(unsignedToDecimal(v)));
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
            int v = decimalToUnsigned(n[j].val);
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
            o = decimalAdd(decimalMul(o, "256"), unsignedToDecimal(vj));
        }
        out.push_back(o);
    }
    return out;
}
