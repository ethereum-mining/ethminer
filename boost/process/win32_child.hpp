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
 * \file boost/process/win32_child.hpp 
 * 
 * Includes the declaration of the win32_child class. 
 */ 

#ifndef BOOST_PROCESS_WIN32_CHILD_HPP 
#define BOOST_PROCESS_WIN32_CHILD_HPP 

#include <boost/process/child.hpp> 
#include <boost/process/detail/file_handle.hpp> 
#include <windows.h> 

namespace boost { 
namespace process { 

/** 
 * Windows implementation of the Child concept. 
 * 
 * The win32_child class implements the Child concept in a Windows 
 * operating system. 
 * 
 * A Windows child differs from a regular %child (represented by a 
 * child object) in that it holds additional information about a process. 
 * Aside from the standard handle, it also includes a handle to the 
 * process' main thread, together with identifiers to both entities. 
 * 
 * This class is built on top of the generic child so as to allow its 
 * trivial adoption. When a program is changed to use the 
 * Windows-specific context (win32_context), it will most certainly need 
 * to migrate its use of the child class to win32_child. Doing so is only 
 * a matter of redefining the appropriate object and later using the 
 * required extra features: there should be no need to modify the existing 
 * code (e.g. method calls) in any other way. 
 */ 
class win32_child : public child 
{ 
public: 
    /** 
     * Constructs a new Windows child object representing a just 
     * spawned %child process. 
     * 
     * Creates a new %child object that represents the process described by 
     * the \a pi structure. 
     * 
     * The \a fhstdin, \a fhstdout and \a fhstderr parameters hold the 
     * communication streams used to interact with the %child process if 
     * the launcher configured redirections. See the parent class' 
     * constructor for more details on these. 
     * 
     * \see child 
     */ 
    win32_child(const PROCESS_INFORMATION &pi, detail::file_handle fhstdin, detail::file_handle fhstdout, detail::file_handle fhstderr) 
        : child(pi.dwProcessId, fhstdin, fhstdout, fhstderr, pi.hProcess), 
        process_information_(pi), 
        thread_handle_(process_information_.hThread) 
    { 
    } 

    /** 
     * Returns the process handle. 
     * 
     * Returns a process-specific handle that can be used to access the 
     * process. This is the value of the \a hProcess field in the 
     * PROCESS_INFORMATION structure returned by CreateProcess(). 
     * 
     * \see get_id() 
     */ 
    HANDLE get_handle() const 
    { 
        return process_information_.hProcess; 
    } 

    /** 
     * Returns the primary thread's handle. 
     * 
     * Returns a handle to the primary thread of the new process. This is 
     * the value of the \a hThread field in the PROCESS_INFORMATION 
     * structure returned by CreateProcess(). 
     * 
     * \see get_primary_thread_id() 
     */ 
    HANDLE get_primary_thread_handle() const 
    { 
        return process_information_.hThread; 
    } 

    /** 
     * Returns the primary thread's identifier. 
     * 
     * Returns a system-wide value that identifies the process's primary 
     * thread. This is the value of the \a dwThreadId field in the 
     * PROCESS_INFORMATION structure returned by CreateProcess(). 
     * 
     * \see get_primary_thread_handle() 
     */ 
    DWORD get_primary_thread_id() const 
    { 
        return process_information_.dwThreadId; 
    } 

private: 
    /** 
     * Windows-specific process information. 
     */ 
    PROCESS_INFORMATION process_information_; 

    /** 
     * Thread handle owned by RAII object. 
     */ 
    detail::file_handle thread_handle_; 
}; 

} 
} 

#endif 
