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
typedef boost::uniform_int<uint64_t> boostUint64Distrib;

typedef boost::random::variate_generator<boost::mt19937&, boostIntDistrib > boostIntGenerator;
typedef boost::random::variate_generator<boost::mt19937&, boostDescreteDistrib > boostWeightGenerator;
typedef boost::random::variate_generator<boost::mt19937&, boostUint64Distrib > boostUInt64Generator;

struct RandomCodeOptions
{
public:
	RandomCodeOptions();
	void setWeight(dev::eth::Instruction _opCode, int _weight);
	void addAddress(dev::Address const& _address);
	dev::Address getRandomAddress() const;

	bool useUndefinedOpCodes;
	int smartCodeProbability;
	boostDescreteDistrib opCodeProbability;
private:
	void setWeights();
	std::map<int, int> mapWeights;
	std::vector<dev::Address> addressList;
};

enum class SizeStrictness
{
	Strict,
	Random
};

struct RlpDebug
{
	std::string rlp;
	int insertions;
};

class RandomCode
{
public:
	/// Generate random vm code
	static std::string generate(int _maxOpNumber = 1, RandomCodeOptions _options = RandomCodeOptions());

	/// Generate random byte string of a given length
	static std::string rndByteSequence(int _length = 1, SizeStrictness _sizeType = SizeStrictness::Strict);

	/// Generate random rlp byte sequence of a given depth (e.g [[[]],[]]). max depth level = 20.
	/// The _debug string contains returned rlp string with analysed sections
	/// [] - length section/ or single byte rlp encoding
	/// () - decimal representation of length
	/// {1} - Array
	/// {2} - Array more than 55
	/// {3} - List
	/// {4} - List more than 55
	static std::string rndRLPSequence(int _depth, std::string& _debug);

	/// Generate random int64
	static std::string randomUniIntHex(u256 _maxVal = 0);
	static int randomUniInt();

private:
	static std::string fillArguments(dev::eth::Instruction _opcode, RandomCodeOptions const& _options);
	static std::string getPushCode(int _value);
	static std::string getPushCode(std::string const& _hex);
	static int recursiveRLP(std::string& _result, int _depth, std::string& _debug);
	static void refreshSeed();

	static boost::random::mt19937 gen;			///< Random generator
	static boostIntDistrib opCodeDist;			///< 0..255 opcodes
	static boostIntDistrib opLengDist;			///< 1..32  byte string
	static boostIntDistrib uniIntDist;          ///< 0..0x7fffffff
	static boostUint64Distrib uInt64Dist;		///< 0..2**64

	static boostIntGenerator randUniIntGen;		///< Generate random UniformInt from uniIntDist
	static boostIntGenerator randOpCodeGen;		///< Generate random value from opCodeDist
	static boostIntGenerator randOpLengGen;		///< Generate random length from opLengDist
	static boostUInt64Generator randUInt64Gen;	///< Generate random uInt64
};

}
}
