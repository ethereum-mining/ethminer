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

static const char* jumpDestName = "JmpDst.";
static const char* basicBlockName = "Instr.";

BasicBlock::BasicBlock(instr_idx _firstInstrIdx, code_iterator _begin, code_iterator _end, llvm::Function* _mainFunc, bool isJumpDest):
	m_firstInstrIdx{_firstInstrIdx},
	m_begin(_begin),
	m_end(_end),
	m_llvmBB(llvm::BasicBlock::Create(_mainFunc->getContext(), {isJumpDest ? jumpDestName : basicBlockName, std::to_string(_firstInstrIdx)}, _mainFunc)),
	m_isJumpDest(isJumpDest)
{}

LocalStack::LocalStack(Stack& _globalStack):
	m_global(_globalStack)
{}

void LocalStack::push(llvm::Value* _value)
{
	assert(_value->getType() == Type::Word);
	m_currentStack.push_back(_value);
	m_maxSize = std::max(m_maxSize, size());
}

llvm::Value* LocalStack::pop()
{
	auto item = get(0);
	assert(!m_currentStack.empty() || !m_initialStack.empty());

	if (m_currentStack.size() > 0)
		m_currentStack.pop_back();
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
	if (_index < m_currentStack.size())
		return *(m_currentStack.rbegin() + _index); // count from back

	auto idx = _index - m_currentStack.size() + m_globalPops;
	if (idx >= m_initialStack.size())
		m_initialStack.resize(idx + 1);
	auto& item = m_initialStack[idx];

	if (!item)
		item = m_global.get(idx);

	return item;
}

void LocalStack::set(size_t _index, llvm::Value* _word)
{
	if (_index < m_currentStack.size())
	{
		*(m_currentStack.rbegin() + _index) = _word;
		return;
	}

	auto idx = _index - m_currentStack.size() + m_globalPops;
	assert(idx < m_initialStack.size());
	m_initialStack[idx] = _word;
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
		assert(m_globalPops <= m_initialStack.size()); // pop() always does get()
		for (auto i = m_globalPops; i < m_initialStack.size(); ++i)
		{
			if (m_initialStack[i])
				m_global.set(i, m_initialStack[i]);
		}

		// Add new items
		for (auto& item: m_currentStack)
		{
			if (m_globalPops) 						// Override poped global items
				m_global.set(--m_globalPops, item);	// using pops counter as the index
			else
				m_global.push(item);
		}

		// Pop not overriden items
		if (m_globalPops)
			m_global.pop(m_globalPops);
	}
}


}
}
}
