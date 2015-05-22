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

boostIntGenerator RandomCode::randOpCodeGen = boostIntGenerator(gen, opCodeDist);
boostIntGenerator RandomCode::randOpLengGen = boostIntGenerator(gen, opLengDist);
boostIntGenerator RandomCode::randUniIntGen = boostIntGenerator(gen, uniIntDist);

std::string RandomCode::rndByteSequence(int length)
{
	refreshSeed();
	std::string hash;
	length = std::max(1, length);
	for (auto i = 0; i < length; i++)
	{
		uint8_t byte = randOpCodeGen();
		hash += toCompactHex(byte);
	}
	return hash;
}

std::string RandomCode::fillArguments(int num)
{
	std::string code;
	for (auto i = 0; i < num; i++)
	{
		int length = randOpLengGen();
		int pushCode = 96 + length - 1;
		code += toCompactHex(pushCode) + rndByteSequence(length);
	}
	return code;
}

//generate smart random code
std::string RandomCode::generate(int maxOpNumber, RandomCodeOptions options)
{
	refreshSeed();
	std::string code;

	//random opCode amount
	boostIntDistrib sizeDist (0, maxOpNumber);
	boostIntGenerator rndSizeGen(gen, sizeDist);
	int size = (int)rndSizeGen();

	boostWeightGenerator randOpCodeWeight (gen, options.opCodeProbability);
	bool weightsDefined = options.opCodeProbability.probabilities().size() == 255;

	for (auto i = 0; i < size; i++)
	{
		uint8_t opcode = weightsDefined ? randOpCodeWeight() : randOpCodeGen();
		dev::eth::InstructionInfo info = dev::eth::instructionInfo((dev::eth::Instruction) opcode);

		if (info.name.find_first_of("INVALID_INSTRUCTION") > 0)
		{
			//Byte code is yet not implemented
			if (options.useUndefinedOpCodes == false)
			{
				i--;
				continue;
			}
		}
		else
			code += fillArguments(info.args);
		std::string byte = toCompactHex(opcode);
		code += (byte == "") ? "00" : byte;
	}
	return code;
}

std::string RandomCode::randomUniInt()
{
	refreshSeed();
	return "0x" + toCompactHex((int)randUniIntGen());
}

void RandomCode::refreshSeed()
{
	auto now = std::chrono::steady_clock::now().time_since_epoch();
	auto timeSinceEpoch = std::chrono::duration_cast<std::chrono::nanoseconds>(now).count();
	gen.seed(static_cast<unsigned int>(timeSinceEpoch));
}

}
}
