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

Node compileToLLL(std::string input) {
    return rewrite(parseSerpent(input));
}

std::string compile(std::string input) {
    return compileLLL(compileToLLL(input));
}

std::vector<Node> prettyCompile(std::string input) {
    return prettyCompileLLL(compileToLLL(input));
}
