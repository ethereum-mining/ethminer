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
/** @file Sealer.h
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 *
 * Determines the PoW algorithm.
 */

#pragma once

#include <functional>
#include <libdevcore/Guards.h>
#include <libdevcore/RLP.h>
#include "Common.h"

namespace dev
{
namespace eth
{

class BlockInfo;

class SealEngineFace
{
public:
	virtual ~SealEngineFace() {}

	virtual std::string name() const = 0;
	virtual unsigned revision() const = 0;
	virtual unsigned sealFields() const = 0;
	virtual bytes sealRLP() const = 0;

	bytes option(std::string const& _name) const { Guard l(x_options); return m_options.count(_name) ? m_options.at(_name) : bytes(); }
	bool setOption(std::string const& _name, bytes const& _value) { Guard l(x_options); try { if (onOptionChanging(_name, _value)) { m_options[_name] = _value; return true; } } catch (...) {} return false; }

	virtual strings sealers() const { return { "default" }; }
	virtual void setSealer(std::string const&) {}
	virtual void generateSeal(BlockInfo const& _bi) = 0;
	virtual void onSealGenerated(std::function<void(bytes const& s)> const& _f) = 0;
	virtual void cancelGeneration() {}

protected:
	virtual bool onOptionChanging(std::string const&, bytes const&) { return true; }
	void injectOption(std::string const& _name, bytes const& _value) { Guard l(x_options); m_options[_name] = _value; }

private:
	mutable Mutex x_options;
	std::unordered_map<std::string, bytes> m_options;
};

template <class Sealer>
class SealEngineBase: public SealEngineFace
{
public:
	std::string name() const override { return Sealer::name(); }
	unsigned revision() const override { return Sealer::revision(); }
	unsigned sealFields() const override { return Sealer::BlockHeader::SealFields; }
	bytes sealRLP() const override { RLPStream s(sealFields()); s.appendRaw(typename Sealer::BlockHeader().sealFieldsRLP(), sealFields()); return s.out(); }
};

}
}
