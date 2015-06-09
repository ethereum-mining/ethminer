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
			code += fillArguments((dev::eth::Instruction) opcode, _options);
		std::string byte = toCompactHex(opcode);
		code += (byte == "") ? "00" : byte;
	}
	return code;
}

std::string RandomCode::randomUniIntHex()
{
	refreshSeed();
	int rand = randUniIntGen() % 100;
	if (rand < 50)
		return "0x" + toCompactHex((u256)randUniIntGen());
	return "0x" + toCompactHex((u256)randUInt64Gen());
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
		case dev::eth::Instruction::CALL:
			//(CALL gaslimit address value memstart1 memlen1 memstart2 memlen2)
			code += getPushCode(randUniIntGen() % 32);  //memlen2
			code += getPushCode(randUniIntGen() % 32);  //memstart2
			code += getPushCode(randUniIntGen() % 32);  //memlen1
			code += getPushCode(randUniIntGen() % 32);  //memlen1
			code += getPushCode(randUniIntGen());		//value
			code += getPushCode(toString(_options.getRandomAddress()));//address
			code += getPushCode(randUniIntGen());		//gaslimit
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
