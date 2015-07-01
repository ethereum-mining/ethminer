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
/** @file fuzzHelper.cpp
 * @author Dimitry Khokhlov <winsvega@mail.ru>
 * @date 2015
 */

#include "fuzzHelper.h"

#include <chrono>
#include <boost/random.hpp>
#include <boost/filesystem/path.hpp>
#include <libevmcore/Instruction.h>

namespace dev
{
namespace test
{

boost::random::mt19937 RandomCode::gen;
boostIntDistrib RandomCode::opCodeDist = boostIntDistrib (0, 255);
boostIntDistrib RandomCode::opLengDist = boostIntDistrib (1, 32);
boostIntDistrib RandomCode::uniIntDist = boostIntDistrib (0, 0x7fffffff);
boostUint64Distrib RandomCode::uInt64Dist = boostUint64Distrib (0, std::numeric_limits<uint64_t>::max());

boostIntGenerator RandomCode::randOpCodeGen = boostIntGenerator(gen, opCodeDist);
boostIntGenerator RandomCode::randOpLengGen = boostIntGenerator(gen, opLengDist);
boostIntGenerator RandomCode::randUniIntGen = boostIntGenerator(gen, uniIntDist);
boostUInt64Generator RandomCode::randUInt64Gen = boostUInt64Generator(gen, uInt64Dist);

std::string RandomCode::rndByteSequence(int _length, SizeStrictness _sizeType)
{
	refreshSeed();
	std::string hash;
	_length = (_sizeType == SizeStrictness::Strict) ? std::max(1, _length) : randomUniInt() % _length;
	for (auto i = 0; i < _length; i++)
	{
		uint8_t byte = randOpCodeGen();
		hash += toCompactHex(byte, HexPrefix::DontAdd, 1);
	}
	return hash;
}

//generate smart random code
std::string RandomCode::generate(int _maxOpNumber, RandomCodeOptions _options)
{
	refreshSeed();
	std::string code;

	//random opCode amount
	boostIntDistrib sizeDist (0, _maxOpNumber);
	boostIntGenerator rndSizeGen(gen, sizeDist);
	int size = (int)rndSizeGen();

	boostWeightGenerator randOpCodeWeight (gen, _options.opCodeProbability);
	bool weightsDefined = _options.opCodeProbability.probabilities().size() == 255;

	for (auto i = 0; i < size; i++)
	{
		uint8_t opcode = weightsDefined ? randOpCodeWeight() : randOpCodeGen();
		dev::eth::InstructionInfo info = dev::eth::instructionInfo((dev::eth::Instruction) opcode);

		if (info.name.find_first_of("INVALID_INSTRUCTION") > 0)
		{
			//Byte code is yet not implemented
			if (_options.useUndefinedOpCodes == false)
			{
				i--;
				continue;
			}
		}
		else
		{
			if (info.name.find_first_of("PUSH") > 0)
				code += toCompactHex(opcode);
			code += fillArguments((dev::eth::Instruction) opcode, _options);
		}

		if (info.name.find_first_of("PUSH") <= 0)
		{
			std::string byte = toCompactHex(opcode);
			code += (byte == "") ? "00" : byte;
		}
	}
	return code;
}

std::string RandomCode::randomUniIntHex(u256 _maxVal)
{
	if (_maxVal == 0)
		_maxVal = std::numeric_limits<uint64_t>::max();
	refreshSeed();
	int rand = randUniIntGen() % 100;
	if (rand < 50)
		return "0x" + toCompactHex((u256)randUniIntGen() % _maxVal);
	return "0x" + toCompactHex((u256)randUInt64Gen() % _maxVal);
}

int RandomCode::randomUniInt()
{
	refreshSeed();
	return (int)randUniIntGen();
}

void RandomCode::refreshSeed()
{
	auto now = std::chrono::steady_clock::now().time_since_epoch();
	auto timeSinceEpoch = std::chrono::duration_cast<std::chrono::nanoseconds>(now).count();
	gen.seed(static_cast<unsigned int>(timeSinceEpoch));
}

std::string RandomCode::getPushCode(std::string const& _hex)
{
	int length = _hex.length() / 2;
	int pushCode = 96 + length - 1;
	return toCompactHex(pushCode) + _hex;
}

std::string RandomCode::getPushCode(int _value)
{
	std::string hexString = toCompactHex(_value);
	return getPushCode(hexString);
}

std::string RandomCode::fillArguments(dev::eth::Instruction _opcode, RandomCodeOptions const& _options)
{
	dev::eth::InstructionInfo info = dev::eth::instructionInfo(_opcode);

	std::string code;
	bool smart = false;
	unsigned num = info.args;
	int rand = randUniIntGen() % 100;
	if (rand < _options.smartCodeProbability)
		smart = true;

	if (smart)
	{
		switch (_opcode)
		{
		case dev::eth::Instruction::PUSH1:	code += rndByteSequence(1);	break;
		case dev::eth::Instruction::PUSH2:	code += rndByteSequence(2);	break;
		case dev::eth::Instruction::PUSH3:	code += rndByteSequence(3);	break;
		case dev::eth::Instruction::PUSH4:	code += rndByteSequence(4);	break;
		case dev::eth::Instruction::PUSH5:	code += rndByteSequence(5);	break;
		case dev::eth::Instruction::PUSH6:	code += rndByteSequence(6);	break;
		case dev::eth::Instruction::PUSH7:	code += rndByteSequence(7);	break;
		case dev::eth::Instruction::PUSH8:	code += rndByteSequence(8);	break;
		case dev::eth::Instruction::PUSH9:	code += rndByteSequence(9);	break;
		case dev::eth::Instruction::PUSH10:	code += rndByteSequence(10); break;
		case dev::eth::Instruction::PUSH11:	code += rndByteSequence(11); break;
		case dev::eth::Instruction::PUSH12:	code += rndByteSequence(12); break;
		case dev::eth::Instruction::PUSH13:	code += rndByteSequence(13); break;
		case dev::eth::Instruction::PUSH14:	code += rndByteSequence(14); break;
		case dev::eth::Instruction::PUSH15:	code += rndByteSequence(15); break;
		case dev::eth::Instruction::PUSH16:	code += rndByteSequence(16); break;
		case dev::eth::Instruction::PUSH17:	code += rndByteSequence(17); break;
		case dev::eth::Instruction::PUSH18:	code += rndByteSequence(18); break;
		case dev::eth::Instruction::PUSH19:	code += rndByteSequence(19); break;
		case dev::eth::Instruction::PUSH20:	code += rndByteSequence(20); break;
		case dev::eth::Instruction::PUSH21:	code += rndByteSequence(21); break;
		case dev::eth::Instruction::PUSH22:	code += rndByteSequence(22); break;
		case dev::eth::Instruction::PUSH23:	code += rndByteSequence(23); break;
		case dev::eth::Instruction::PUSH24:	code += rndByteSequence(24); break;
		case dev::eth::Instruction::PUSH25:	code += rndByteSequence(25); break;
		case dev::eth::Instruction::PUSH26:	code += rndByteSequence(26); break;
		case dev::eth::Instruction::PUSH27:	code += rndByteSequence(27); break;
		case dev::eth::Instruction::PUSH28:	code += rndByteSequence(28); break;
		case dev::eth::Instruction::PUSH29:	code += rndByteSequence(29); break;
		case dev::eth::Instruction::PUSH30:	code += rndByteSequence(30); break;
		case dev::eth::Instruction::PUSH31:	code += rndByteSequence(31); break;
		case dev::eth::Instruction::PUSH32:	code += rndByteSequence(32); break;
		case dev::eth::Instruction::SWAP1:
		case dev::eth::Instruction::SWAP2:
		case dev::eth::Instruction::SWAP3:
		case dev::eth::Instruction::SWAP4:
		case dev::eth::Instruction::SWAP5:
		case dev::eth::Instruction::SWAP6:
		case dev::eth::Instruction::SWAP7:
		case dev::eth::Instruction::SWAP8:
		case dev::eth::Instruction::SWAP9:
		case dev::eth::Instruction::SWAP10:
		case dev::eth::Instruction::SWAP11:
		case dev::eth::Instruction::SWAP12:
		case dev::eth::Instruction::SWAP13:
		case dev::eth::Instruction::SWAP14:
		case dev::eth::Instruction::SWAP15:
		case dev::eth::Instruction::SWAP16:
		case dev::eth::Instruction::DUP1:
		case dev::eth::Instruction::DUP2:
		case dev::eth::Instruction::DUP3:
		case dev::eth::Instruction::DUP4:
		case dev::eth::Instruction::DUP5:
		case dev::eth::Instruction::DUP6:
		case dev::eth::Instruction::DUP7:
		case dev::eth::Instruction::DUP8:
		case dev::eth::Instruction::DUP9:
		case dev::eth::Instruction::DUP10:
		case dev::eth::Instruction::DUP11:
		case dev::eth::Instruction::DUP12:
		case dev::eth::Instruction::DUP13:
		case dev::eth::Instruction::DUP14:
		case dev::eth::Instruction::DUP15:
		case dev::eth::Instruction::DUP16:
			int times;
			switch (_opcode)
			{
				case dev::eth::Instruction::DUP1:	times = 1; break;
				case dev::eth::Instruction::SWAP1:
				case dev::eth::Instruction::DUP2:	times = 2; break;
				case dev::eth::Instruction::SWAP2:
				case dev::eth::Instruction::DUP3:	times = 3; break;
				case dev::eth::Instruction::SWAP3:
				case dev::eth::Instruction::DUP4:	times = 4; break;
				case dev::eth::Instruction::SWAP4:
				case dev::eth::Instruction::DUP5:	times = 5; break;
				case dev::eth::Instruction::SWAP5:
				case dev::eth::Instruction::DUP6:	times = 6; break;
				case dev::eth::Instruction::SWAP6:
				case dev::eth::Instruction::DUP7:	times = 7; break;
				case dev::eth::Instruction::SWAP7:
				case dev::eth::Instruction::DUP8:	times = 8; break;
				case dev::eth::Instruction::SWAP8:
				case dev::eth::Instruction::DUP9:	times = 9; break;
				case dev::eth::Instruction::SWAP9:
				case dev::eth::Instruction::DUP10:	times = 10; break;
				case dev::eth::Instruction::SWAP10:
				case dev::eth::Instruction::DUP11:	times = 11; break;
				case dev::eth::Instruction::SWAP11:
				case dev::eth::Instruction::DUP12:	times = 12; break;
				case dev::eth::Instruction::SWAP12:
				case dev::eth::Instruction::DUP13:	times = 13; break;
				case dev::eth::Instruction::SWAP13:
				case dev::eth::Instruction::DUP14:	times = 14; break;
				case dev::eth::Instruction::SWAP14:
				case dev::eth::Instruction::DUP15:	times = 15; break;
				case dev::eth::Instruction::SWAP15:
				case dev::eth::Instruction::DUP16:	times = 16; break;
				case dev::eth::Instruction::SWAP16:	times = 17; break;
				default: times = 1;
			}
			for (int i = 0; i < times; i ++)
				code += getPushCode(randUniIntGen() % 32);

		break;
		case dev::eth::Instruction::CREATE:
			//(CREATE value mem1 mem2)
			code += getPushCode(randUniIntGen() % 128);  //memlen1
			code += getPushCode(randUniIntGen() % 32);   //memlen1
			code += getPushCode(randUniIntGen());		 //value
		break;
		case dev::eth::Instruction::CALL:
		case dev::eth::Instruction::CALLCODE:
			//(CALL gaslimit address value memstart1 memlen1 memstart2 memlen2)
			//(CALLCODE gaslimit address value memstart1 memlen1 memstart2 memlen2)
			code += getPushCode(randUniIntGen() % 128);  //memlen2
			code += getPushCode(randUniIntGen() % 32);  //memstart2
			code += getPushCode(randUniIntGen() % 128);  //memlen1
			code += getPushCode(randUniIntGen() % 32);  //memlen1
			code += getPushCode(randUniIntGen());		//value
			code += getPushCode(toString(_options.getRandomAddress()));//address
			code += getPushCode(randUniIntGen());		//gaslimit
		break;
		case dev::eth::Instruction::SUICIDE: //(SUICIDE address)
			code += getPushCode(toString(_options.getRandomAddress()));
		break;
		case dev::eth::Instruction::RETURN:  //(RETURN memlen1 memlen2)
			code += getPushCode(randUniIntGen() % 128);  //memlen1
			code += getPushCode(randUniIntGen() % 32);  //memlen1
		break;
		default:
			smart = false;
		}
	}

	if (smart == false)
	for (unsigned i = 0; i < num; i++)
	{
		//generate random parameters
		int length = randOpLengGen();
		code += getPushCode(rndByteSequence(length));
	}
	return code;
}


//Ramdom Code Options
RandomCodeOptions::RandomCodeOptions() : useUndefinedOpCodes(false), smartCodeProbability(50)
{
	//each op code with same weight-probability
	for (auto i = 0; i < 255; i++)
		mapWeights.insert(std::pair<int, int>(i, 50));
	setWeights();
}

void RandomCodeOptions::setWeight(dev::eth::Instruction _opCode, int _weight)
{
	mapWeights.at((int)_opCode) = _weight;
	setWeights();
}

void RandomCodeOptions::addAddress(dev::Address const& _address)
{
	addressList.push_back(_address);
}

dev::Address RandomCodeOptions::getRandomAddress() const
{
	if (addressList.size() > 0)
	{
		int index = RandomCode::randomUniInt() % addressList.size();
		return addressList[index];
	}
	return Address(RandomCode::rndByteSequence(20));
}

void RandomCodeOptions::setWeights()
{
	std::vector<int> weights;
	for (auto const& element: mapWeights)
		weights.push_back(element.second);
	opCodeProbability = boostDescreteDistrib(weights);
}


BOOST_AUTO_TEST_SUITE(RandomCodeTests)

BOOST_AUTO_TEST_CASE(rndCode)
{
	std::string code;
	std::cerr << "Testing Random Code: ";
	try
	{
		code = dev::test::RandomCode::generate(10);
	}
	catch(...)
	{
		BOOST_ERROR("Exception thrown when generating random code!");
	}
	std::cerr << code;
}

BOOST_AUTO_TEST_SUITE_END()

}
}
