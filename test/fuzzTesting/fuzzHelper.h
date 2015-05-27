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
/** @file fuzzHelper.h
 * @author Dimitry Khokhlov <winsvega@mail.ru>
 * @date 2015
 */

#include <string>
#include <boost/random.hpp>
#include <boost/filesystem/path.hpp>

#include <test/TestHelper.h>
#include <libdevcore/CommonIO.h>
#include <libdevcore/CommonData.h>
#include <libevmcore/Instruction.h>

#pragma once

namespace dev
{
namespace test
{

typedef boost::random::uniform_int_distribution<> boostIntDistrib;
typedef boost::random::discrete_distribution<> boostDescreteDistrib;

typedef boost::random::variate_generator<boost::mt19937&, boostIntDistrib > boostIntGenerator;
typedef boost::random::variate_generator<boost::mt19937&, boostDescreteDistrib > boostWeightGenerator;

struct RandomCodeOptions
{
public:
	RandomCodeOptions() : useUndefinedOpCodes(false) {
		//each op code with same weight-probability
		for (auto i = 0; i < 255; i++)
			mapWeights.insert(std::pair<int, int>(i, 50));
		setWeights();
	}
	void setWeight(dev::eth::Instruction opCode, int weight)
	{
		mapWeights.at((int)opCode) = weight;
		setWeights();
	}
	bool useUndefinedOpCodes;
	boostDescreteDistrib opCodeProbability;
private:
	void setWeights()
	{
		std::vector<int> weights;
		for (auto const& element: mapWeights)
			weights.push_back(element.second);
		opCodeProbability = boostDescreteDistrib(weights);
	}
	std::map<int, int> mapWeights;
};

class RandomCode
{
public:
	/// Generate random vm code
	static std::string generate(int maxOpNumber = 1, RandomCodeOptions options = RandomCodeOptions());

	/// Generate random byte string of a given length
	static std::string rndByteSequence(int length = 1);

	/// Generate random uniForm Int with reasonable value 0..0x7fffffff
	static std::string randomUniInt();

private:
	static std::string fillArguments(int num);
	static void refreshSeed();

	static boost::random::mt19937 gen;			///< Random generator
	static boostIntDistrib opCodeDist;			///< 0..255 opcodes
	static boostIntDistrib opLengDist;			///< 1..32  byte string
	static boostIntDistrib uniIntDist;          ///< 0..0x7fffffff

	static boostIntGenerator randUniIntGen;		///< Generate random UniformInt from uniIntDist
	static boostIntGenerator randOpCodeGen;		///< Generate random value from opCodeDist
	static boostIntGenerator randOpLengGen;		///< Generate random length from opLengDist
};

}
}
