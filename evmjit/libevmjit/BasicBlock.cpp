#include "BasicBlock.h"

#include <iostream>

#include "preprocessor/llvm_includes_start.h"
#include <llvm/IR/CFG.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/Support/raw_os_ostream.h>
#include "preprocessor/llvm_includes_end.h"

#include "Type.h"
#include "Utils.h"

namespace dev
{
namespace eth
{
namespace jit
{

BasicBlock::BasicBlock(instr_idx _firstInstrIdx, code_iterator _begin, code_iterator _end, llvm::Function* _mainFunc):
	m_firstInstrIdx{_firstInstrIdx},
	m_begin(_begin),
	m_end(_end),
	m_llvmBB(llvm::BasicBlock::Create(_mainFunc->getContext(), {"Instr.", std::to_string(_firstInstrIdx)}, _mainFunc))
{}

LocalStack::LocalStack(Stack& _globalStack):
	m_global(_globalStack)
{}

void LocalStack::push(llvm::Value* _value)
{
	assert(_value->getType() == Type::Word);
	m_local.push_back(_value);
	m_maxSize = std::max(m_maxSize, size());
}

llvm::Value* LocalStack::pop()
{
	auto item = get(0);
	assert(!m_local.empty() || !m_input.empty());

	if (m_local.size() > 0)
		m_local.pop_back();
	else
		++m_globalPops;

	m_minSize = std::min(m_minSize, size());
	return item;
}

/**
 *  Pushes a copy of _index-th element (tos is 0-th elem).
 */
void LocalStack::dup(size_t _index)
{
	auto val = get(_index);
	push(val);
}

/**
 *  Swaps tos with _index-th element (tos is 0-th elem).
 *  _index must be > 0.
 */
void LocalStack::swap(size_t _index)
{
	assert(_index > 0);
	auto val = get(_index);
	auto tos = get(0);
	set(_index, tos);
	set(0, val);
}

llvm::Value* LocalStack::get(size_t _index)
{
	if (_index < m_local.size())
		return *(m_local.rbegin() + _index); // count from back

	auto idx = _index - m_local.size() + m_globalPops;
	if (idx >= m_input.size())
		m_input.resize(idx + 1);
	auto& item = m_input[idx];

	if (!item)
	{
		item = m_global.get(idx); 											// Reach an item from global stack
		m_minSize = std::min(m_minSize, -static_cast<ssize_t>(idx) - 1); 	// and remember required stack size
	}

	return item;
}

void LocalStack::set(size_t _index, llvm::Value* _word)
{
	if (_index < m_local.size())
	{
		*(m_local.rbegin() + _index) = _word;
		return;
	}

	auto idx = _index - m_local.size() + m_globalPops;
	assert(idx < m_input.size());
	m_input[idx] = _word;
}


void LocalStack::finalize(llvm::IRBuilder<>& _builder, llvm::BasicBlock& _bb)
{
	auto blockTerminator = _bb.getTerminator();
	assert(blockTerminator);
	if (blockTerminator->getOpcode() != llvm::Instruction::Ret)
	{
		// Not needed in case of ret instruction. Ret invalidates the stack.
		_builder.SetInsertPoint(blockTerminator);

		// Update items fetched from global stack ignoring the poped ones
		assert(m_globalPops <= m_input.size()); // pop() always does get()
		for (auto i = m_globalPops; i < m_input.size(); ++i)
		{
			if (m_input[i])
				m_global.set(i, m_input[i]);
		}

		// Add new items
		auto pops = m_globalPops;			// Copy pops counter to keep original value
		for (auto& item: m_local)
		{
			if (pops) 						// Override poped global items
				m_global.set(--pops, item);	// using pops counter as the index
			else
				m_global.push(item);
		}

		// Pop not overriden items
		if (pops)
			m_global.pop(pops);
	}
}


}
}
}
