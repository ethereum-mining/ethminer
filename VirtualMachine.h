#pragma once

#include <map>
#include <unordered_map>
#include "Common.h"

namespace eth
{

enum class Instruction: uint8_t
{
	Stop = 0x00, //halts execution
	Add = 0x10, // Rx Ry Rz - sets Rz <- Rx + Ry mod 2^256
/*	(11) SUB Rx Ry Rz - sets Rz <- Rx - Ry mod 2^256
	(12) MUL Rx Ry Rz - sets Rz <- Rx * Ry mod 2^256
	(13) DIV Rx Ry Rz - sets Rz <- floor(Rx / Ry)
	(14) SDIV Rx Ry Rz - like DIV, except it treats values above 2^255 as negative (ie. 2^256 - x -> -x)
	(15) MOD Rx Ry Rz - sets Rz <- Rx mod Ry
	(16) SMOD Rx Ry Rz - like MOD, but for signed values just like SDIV (using Python's convention with negative numbers)
	(17) EXP Rx Ry Rz - sets Rz <- Rx ^ Ry mod 2^256
	(18) NEG Rx Ry - sets Ry <- 2^256 - Rx
	(20) LT Rx Ry Rz - sets Rz <- 1 if Rx < Ry else 0
	(21) LE Rx Ry Rz - sets Rz <- 1 if Rx <= Ry else 0
	(22) GT Rx Ry Rz - sets Rz <- 1 if Rx > Ry else 0
	(23) GE Rx Ry Rz - sets Rz <- 1 if Rx >= Ry else 0
	(24) EQ Rx Ry Rz - sets Rz <- 1 if Rx = Ry else 0
	(25) NOT Rx Ry - sets Ry <- 1 if Rx = 0 else 0
	(30) SHA256 Rx Ry - sets Ry <- SHA256(Rx)
	(31) RIPEMD160 Rx Ry - sets Ry <- RIPEMD160(Rx)
	(32) ECMUL Rx Ry Rz Ra Rb - sets (Ra, Rb) = Rz * (Rx, Ry) in secp256k1, using (0,0) for the point at infinity
	(33) ECADD Rx Ry Rz Ra Rb Rc - sets (Rb, Rc) = (Rx, Ry) + (Ra, Rb)
	(34) SIGN Rx Ry Rz Ra Rb - sets(Rz, Ra, Rb)as the(r,s,prefix)values of an Electrum-style RFC6979 deterministic signature ofRxwith private keyRy`
	(35) RECOVER Rx Ry Rz Ra Rb Rc - sets(Rb, Rc)as the public key from the signature(Ry, Rz, Ra)of the message hashRx`
	(40) COPY Rx Ry - copies Ry <- Rx
	(41) ST Rx Ry - sets M[Ry] <- Rx
	(42) LD Rx Ry - sets Ry <- M[Rx]
	(43) SET Rx V1 V2 V3 V4 - sets Rx <- V1 + 2^8*V2 + 2^16*V3 + 2^24*V4 (where 0 <= V[i] <= 255)
	(50) JMP Rx - sets the index pointer to the value at Rx
	(51) JMPI Rx Ry - if Rx != 0, sets the index pointer to Ry
	(52) IND Rx - sets Rx to the index pointer.
	(60) EXTRO Rx Ry Rz - looks at the contract at address Rx and its memory state Ry, and outputs the result to Rz
	(61) BALANCE Rx - returns the ether balance of address Rx
	(70) MKTX Rx Ry Rz Rw Rv - sends Ry ether to Rx plus Rz fee with Rw data items starting from memory index Rv (and then reading to (Rv + 1), (Rv + 2) etc). Note that if Rx = 0 then this creates a new contract.
	(80) DATA Rx Ry - sets Ry to data item index Rx if possible, otherwise zero
	(81) DATAN Rx - sets Rx to the number of data items
	(90) MYADDRESS Rx - sets Rx to the contract's own address*/
	Suicide = 0xff //Rx - destroys the contract and clears all memory, sending the entire balance plus the negative fee from clearing memory minus TXFEE to the address
};

class VirtualMachine
{
public:
	VirtualMachine();
	~VirtualMachine();
	
private:
	std::map<bigint, bigint> m_memory;
};

}


