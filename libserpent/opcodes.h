#ifndef ETHSERP_OPCODES
#define ETHSERP_OPCODES

#include <stdio.h>
#include <iostream>
#include <vector>
#include <map>

std::map<std::string, int> opcodes;
std::map<int, std::string> reverseOpcodes;

// Fetches everything EXCEPT PUSH1..32
std::pair<std::string, int> _opcode(std::string ops, int opi) {
    if (!opcodes.size()) {
        opcodes["STOP"] = 0x00;
        opcodes["ADD"] = 0x01;
        opcodes["MUL"] = 0x02;
        opcodes["SUB"] = 0x03;
        opcodes["DIV"] = 0x04;
        opcodes["SDIV"] = 0x05;
        opcodes["MOD"] = 0x06;
        opcodes["SMOD"] = 0x07;
        opcodes["EXP"] = 0x08;
        opcodes["NEG"] = 0x09;
        opcodes["LT"] = 0x0a;
        opcodes["GT"] = 0x0b;
        opcodes["SLT"] = 0x0c;
        opcodes["SGT"] = 0x0d;
        opcodes["EQ"] = 0x0e; 
        opcodes["NOT"] = 0x0f;
        opcodes["AND"] = 0x10;
        opcodes["OR"] = 0x11;
        opcodes["XOR"] = 0x12;
        opcodes["BYTE"] = 0x13;
        opcodes["SHA3"] = 0x20;
        opcodes["ADDRESS"] = 0x30;
        opcodes["BALANCE"] = 0x31;
        opcodes["ORIGIN"] = 0x32;
        opcodes["CALLER"] = 0x33;
        opcodes["CALLVALUE"] = 0x34;
        opcodes["CALLDATALOAD"] = 0x35;
        opcodes["CALLDATASIZE"] = 0x36;
        opcodes["CALLDATACOPY"] = 0x37;
        opcodes["CODESIZE"] = 0x38;
        opcodes["CODECOPY"] = 0x39;
        opcodes["GASPRICE"] = 0x3a;
        opcodes["PREVHASH"] = 0x40;
        opcodes["COINBASE"] = 0x41;
        opcodes["TIMESTAMP"] = 0x42;
        opcodes["NUMBER"] = 0x43;
        opcodes["DIFFICULTY"] = 0x44;
        opcodes["GASLIMIT"] = 0x45;
        opcodes["POP"] = 0x50; 
        opcodes["DUP"] = 0x51; 
        opcodes["SWAP"] = 0x52;
        opcodes["MLOAD"] = 0x53;
        opcodes["MSTORE"] = 0x54;
        opcodes["MSTORE8"] = 0x55;
        opcodes["SLOAD"] = 0x56;
        opcodes["SSTORE"] = 0x57;
        opcodes["JUMP"] = 0x58;
        opcodes["JUMPI"] = 0x59;
        opcodes["PC"] = 0x5a;
        opcodes["MSIZE"] = 0x5b;
        opcodes["GAS"] = 0x5c;
        opcodes["CREATE"] = 0xf0;
        opcodes["CALL"] = 0xf1;
        opcodes["RETURN"] = 0xf2;
        opcodes["SUICIDE"] = 0xff;
        for (std::map<std::string, int>::iterator it=opcodes.begin();
             it != opcodes.end();
             it++) {
            reverseOpcodes[(*it).second] = (*it).first;
        }
    }
    std::string op;
    int opcode;
    op = reverseOpcodes.count(opi) ? reverseOpcodes[opi] : "";
    opcode = opcodes.count(ops) ? opcodes[ops] : -1;
    return std::pair<std::string, int>(op, opcode);
}

int opcode(std::string op) {
	return _opcode(op, 0).second;
}

std::string op(int opcode) {
	return _opcode("", opcode).first;
}

#endif
