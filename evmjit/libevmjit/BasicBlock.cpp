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

BasicBlock::BasicBlock(instr_idx _firstInstrIdx, code_iterator _begin, code_iterator _end, llvm::Function* _mainFunc, llvm::IRBuilder<>& _builder, bool isJumpDest) :
	m_firstInstrIdx{_firstInstrIdx},
	m_begin(_begin),
	m_end(_end),
	m_llvmBB(llvm::BasicBlock::Create(_mainFunc->getContext(), {isJumpDest ? jumpDestName : basicBlockName, std::to_string(_firstInstrIdx)}, _mainFunc)),
	m_builder(_builder),
	m_isJumpDest(isJumpDest)
{}

BasicBlock::BasicBlock(std::string _name, llvm::Function* _mainFunc, llvm::IRBuilder<>& _builder, bool isJumpDest) :
	m_llvmBB(llvm::BasicBlock::Create(_mainFunc->getContext(), _name, _mainFunc)),
	m_builder(_builder),
	m_isJumpDest(isJumpDest)
{}

LocalStack::LocalStack(BasicBlock& _owner, Stack& _globalStack) :
	m_bblock(_owner),
	m_global(_globalStack)
{}

void LocalStack::push(llvm::Value* _value)
{
	assert(_value->getType() == Type::Word);
	m_bblock.m_currentStack.push_back(_value);
	m_maxSize = std::max(m_maxSize, m_bblock.m_currentStack.size());
}

llvm::Value* LocalStack::pop()
{
	auto item = get(0);
	assert(!m_bblock.m_currentStack.empty() || !m_bblock.m_initialStack.empty());

	if (m_bblock.m_currentStack.size() > 0)
		m_bblock.m_currentStack.pop_back();
	else
		++m_bblock.m_globalPops;

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
	auto& currentStack = m_bblock.m_currentStack;
	if (_index < currentStack.size())
		return *(currentStack.rbegin() + _index); // count from back

	auto& initialStack = m_bblock.m_initialStack;
	auto idx = _index - currentStack.size() + m_bblock.m_globalPops;
	if (idx >= initialStack.size())
		initialStack.resize(idx + 1);
	auto& item = initialStack[idx];

	if (!item)
		item = m_global.get(idx);

	return item;
}

void LocalStack::set(size_t _index, llvm::Value* _word)
{
	auto& currentStack = m_bblock.m_currentStack;
	if (_index < currentStack.size())
	{
		*(currentStack.rbegin() + _index) = _word;
		return;
	}

	auto& initialStack = m_bblock.m_initialStack;
	auto idx = _index - currentStack.size() + m_bblock.m_globalPops;
	assert(idx < initialStack.size());
	initialStack[idx] = _word;
}


void BasicBlock::synchronizeLocalStack(Stack& _evmStack)
{
	auto blockTerminator = m_llvmBB->getTerminator();
	assert(blockTerminator);
	if (blockTerminator->getOpcode() != llvm::Instruction::Ret)
	{
		// Not needed in case of ret instruction. Ret invalidates the stack.
		m_builder.SetInsertPoint(blockTerminator);

		// Update items fetched from global stack ignoring the poped ones
		assert(m_globalPops <= m_initialStack.size()); // pop() always does get()
		for (auto i = m_globalPops; i < m_initialStack.size(); ++i)
		{
			if (m_initialStack[i])
				_evmStack.set(i, m_initialStack[i]);
		}

		// Add new items
		for (auto& item: m_currentStack)
		{
			if (m_globalPops) 							// Override poped global items
				_evmStack.set(--m_globalPops, item);	// using pops counter as the index
			else
				_evmStack.push(item);
		}

		// Pop not overriden items
		if (m_globalPops)
			_evmStack.pop(m_globalPops);
	}
}

void BasicBlock::dump()
{
	dump(std::cerr, false);
}

void BasicBlock::dump(std::ostream& _out, bool _dotOutput)
{
	llvm::raw_os_ostream out(_out);

	out << (_dotOutput ? "" : "Initial stack:\n");
	for (auto val : m_initialStack)
	{
		if (val == nullptr)
			out << "  ?";
		else if (llvm::isa<llvm::ExtractValueInst>(val))
			out << "  " << val->getName();
		else if (llvm::isa<llvm::Instruction>(val))
			out << *val;
		else
			out << "  " << *val;

		out << (_dotOutput ? "\\l" : "\n");
	}

	out << (_dotOutput ? "| " : "Instructions:\n");
	for (auto ins = m_llvmBB->begin(); ins != m_llvmBB->end(); ++ins)
		out << *ins << (_dotOutput ? "\\l" : "\n");

	if (! _dotOutput)
		out << "Current stack:\n";
	else
		out << "|";

	for (auto val = m_currentStack.rbegin(); val != m_currentStack.rend(); ++val)
	{
		if (*val == nullptr)
			out << "  ?";
		else if (llvm::isa<llvm::ExtractValueInst>(*val))
			out << "  " << (*val)->getName();
		else if (llvm::isa<llvm::Instruction>(*val))
			out << **val;
		else
			out << "  " << **val;
		out << (_dotOutput ? "\\l" : "\n");
	}

	if (! _dotOutput)
		out << "  ...\n----------------------------------------\n";
}




}
}
}
