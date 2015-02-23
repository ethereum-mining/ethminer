#pragma once

#include <iostream>

#include <llvm/Support/Debug.h>

// The same as assert, but expression is always evaluated and result returned
#define CHECK(expr) (assert(expr), expr)

// FIXME: Disable for NDEBUG mode
#define DLOG(CHANNEL) !(llvm::DebugFlag && llvm::isCurrentDebugType(#CHANNEL)) ? (void)0 : std::cerr
