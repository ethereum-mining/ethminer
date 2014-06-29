#include <stdio.h>
#include <iostream>
#include <vector>
#include "funcs.h"
#include "bignum.h"
#include "util.h"
#include "parser.h"
#include "lllparser.h"
#include "compiler.h"
#include "rewriter.h"
#include "tokenize.h"
#include <liblll/Compiler.h>
#include <libethential/Common.h>

Node compileToLLL(std::string input) {
    return rewrite(parseSerpent(input));
}

std::vector<uint8_t> compile(std::string input) {
    return eth::compileLLL(printSimple(compileToLLL(input)));
}

std::vector<Node> prettyCompile(std::string input) {
    return deserialize(bytesToString(
            eth::compileLLL(printSimple(compileToLLL(input)))));
}

std::string bytesToString(std::vector<uint8_t> input) {
    std::string o;
    for (unsigned i = 0; i < input.size(); i++) o += (char)input[i];
    return o;
}

std::string bytesToHex(std::vector<uint8_t> input) {
    return binToHex(bytesToString(input));
}
