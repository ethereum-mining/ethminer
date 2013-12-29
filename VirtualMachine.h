#pragma once

#include <array>
#include <map>
#include <unordered_map>
#include "RLP.h"
#include "Common.h"

namespace eth
{

/// Virtual machine bytecode instruction.
enum class Instruction: uint8_t
{
	STOP = 0x00,		///< halts execution
	ADD = 0x10,			///< Rx Ry Rz - sets Rz <- Rx + Ry mod 2^256
	SUB,				///< Rx Ry Rz - sets Rz <- Rx - Ry mod 2^256
	MUL,				///< Rx Ry Rz - sets Rz <- Rx * Ry mod 2^256
	DIV,				///< Rx Ry Rz - sets Rz <- floor(Rx / Ry)
	SDIV,				///< Rx Ry Rz - like DIV, except it treats values above 2^255 as negative (ie. 2^256 - x -> -x)
	MOD,				///< Rx Ry Rz - sets Rz <- Rx mod Ry
	SMOD,				///< Rx Ry Rz - like MOD, but for signed values just like SDIV (using Python's convention with negative numbers)
	EXP,				///< Rx Ry Rz - sets Rz <- Rx ^ Ry mod 2^256
	NEG,				///< Rx Ry - sets Ry <- 2^256 - Rx
	LT = 0x20,			///< Rx Ry Rz - sets Rz <- 1 if Rx < Ry else 0
	LE,					///< Rx Ry Rz - sets Rz <- 1 if Rx <= Ry else 0
	GT,					///< Rx Ry Rz - sets Rz <- 1 if Rx > Ry else 0
	GE,					///< Rx Ry Rz - sets Rz <- 1 if Rx >= Ry else 0
	EQ,					///< Rx Ry Rz - sets Rz <- 1 if Rx = Ry else 0
	NOT,				///< Rx Ry - sets Ry <- 1 if Rx = 0 else 0
	SHA256 = 0x30,		///< Rx Ry - sets Ry <- SHA256(Rx)
	RIPEMD160,			///< Rx Ry - sets Ry <- RIPEMD160(Rx)
	ECMUL,				///< Rx Ry Rz Ra Rb - sets (Ra, Rb) = Rz * (Rx, Ry) in secp256k1, using (0,0) for the point at infinity
	ECADD,				///< Rx Ry Rz Ra Rb Rc - sets (Rb, Rc) = (Rx, Ry) + (Ra, Rb)
	SIGN,				///< Rx Ry Rz Ra Rb - sets(Rz, Ra, Rb)as the(r,s,prefix)values of an Electrum-style RFC6979 deterministic signature ofRxwith private keyRy`
	RECOVER,			///< Rx Ry Rz Ra Rb Rc - sets(Rb, Rc)as the public key from the signature(Ry, Rz, Ra)of the message hashRx`
	COPY = 0x40,		///< Rx Ry - copies Ry <- Rx
	ST,					///< Rx Ry - sets M[Ry] <- Rx
	LD,					///< Rx Ry - sets Ry <- M[Rx]
	SET,				///< Rx V1 V2 V3 V4 - sets Rx <- V1 + 2^8*V2 + 2^16*V3 + 2^24*V4 (where 0 <= V[i] <= 255)
	JMP = 0x50,			///< Rx - sets the index pointer to the value at Rx
	JMPI,				///< Rx Ry - if Rx != 0, sets the index pointer to Ry
	IND,				///< Rx - sets Rx to the index pointer.
	EXTRO = 0x60,		///< Rx Ry Rz - looks at the contract at address Rx and its memory state Ry, and outputs the result to Rz
	BALANCE,			///< Rx - returns the ether balance of address Rx
	MKTX = 0x70,		///< Rx Ry Rz Rw Rv - sends Ry ether to Rx plus Rz fee with Rw data items starting from memory index Rv (and then reading to (Rv + 1), (Rv + 2) etc). Note that if Rx = 0 then this creates a new contract.
	DATA = 0x80,		///< Rx Ry - sets Ry to data item index Rx if possible, otherwise zero
	DATAN,				///< Rx - sets Rx to the number of data items
	MYADDRESS = 0x90,	///< Rx - sets Rx to the contract's own address
	SUICIDE = 0xff		///< Rx - destroys the contract and clears all memory, sending the entire balance plus the negative fee from clearing memory minus TXFEE to the address
};

class VirtualMachine
{
public:
	VirtualMachine();
	~VirtualMachine();

	void initMemory(RLP _contract);
	void setMemory(RLP _state);
	
private:
	std::map<u256, u256> m_memory;
	std::array<u256, 256> m_registers;
	bigint m_stepCount;
	bigint m_totalFee;
	bigint m_stepFee;
	bigint m_dataFee;
	bigint m_memoryFee;
	bigint m_extroFee;
};

}


