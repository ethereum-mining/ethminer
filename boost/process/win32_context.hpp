// 
// Boost.Process 
// ~~~~~~~~~~~~~ 
// 
// Copyright (c) 2006, 2007 Julio M. Merino Vidal 
// Copyright (c) 2008, 2009 Boris Schaeling 
// 
// Distributed under the Boost Software License, Version 1.0. (See accompanying 
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt) 
// 

/** 
 * \file boost/process/win_32context.hpp 
 * 
 * Includes the declaration of the win32_context class. 
 */ 

#ifndef BOOST_PROCESS_WIN32_CONTEXT_HPP 
#define BOOST_PROCESS_WIN32_CONTEXT_HPP 

#include <boost/process/context.hpp> 
#include <string> 
#include <windows.h> 

namespace boost { 
namespace process { 

/** 
 * Generic implementation of the Context concept. 
 * 
 * The context class implements the Context concept in an operating 
 * system agnostic way; it allows spawning new child processes using 
 * a single and common interface across different systems. 
 */ 
template <class String> 
class win32_basic_context : public basic_context<String> 
{ 
public: 
    /** 
     * Initializes the Win32-specific process startup information with NULL. 
     */ 
    win32_basic_context() 
        : startupinfo(NULL) 
    { 
    } 

    /** 
     * Win32-specific process startup information. 
     */ 
    STARTUPINFOA *startupinfo; 
}; 

/** 
 * Default instantiation of win32_basic_context. 
 */ 
typedef win32_basic_context<std::string> win32_context; 

} 
} 

#endif 
