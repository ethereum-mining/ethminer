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
 * \file boost/process/process.hpp 
 * 
 * Includes the declaration of the process class. 
 */ 

#ifndef BOOST_PROCESS_PROCESS_HPP 
#define BOOST_PROCESS_PROCESS_HPP 

#include <boost/process/config.hpp> 

#if defined(BOOST_POSIX_API) 
#  include <cerrno> 
#  include <signal.h> 
#elif defined(BOOST_WINDOWS_API) 
#  include <cstdlib> 
#  include <windows.h> 
#else 
#  error "Unsupported platform." 
#endif 

#include <boost/system/system_error.hpp> 
#include <boost/throw_exception.hpp> 

namespace boost { 
namespace process { 

/** 
 * Generic implementation of the Process concept. 
 * 
 * The process class implements the Process concept in an operating system 
 * agnostic way. 
 */ 
class process 
{ 
public: 
#if defined(BOOST_PROCESS_DOXYGEN) 
    /** 
     * Opaque name for the native process' identifier type. 
     * 
     * Each operating system identifies processes using a specific type. 
     * The \a id_type type is used to transparently refer to a process 
     * regardless of the operating system in which this class is used. 
     * 
     * This type is guaranteed to be an integral type on all supported 
     * platforms. 
     */ 
    typedef NativeProcessId id_type; 
#elif defined(BOOST_POSIX_API) 
    typedef pid_t id_type; 
#elif defined(BOOST_WINDOWS_API) 
    typedef DWORD id_type; 
#endif 

    /** 
     * Constructs a new process object. 
     * 
     * Creates a new process object that represents a running process 
     * within the system. 
     */ 
    process(id_type id) 
        : id_(id) 
    { 
    } 

    /** 
     * Returns the process' identifier. 
     */ 
    id_type get_id() const 
    { 
        return id_; 
    } 

    /** 
     * Terminates the process execution. 
     * 
     * Forces the termination of the process execution. Some platforms 
     * allow processes to ignore some external termination notifications 
     * or to capture them for a proper exit cleanup. You can set the 
     * \a force flag to true in them to force their termination regardless 
     * of any exit handler. 
     * 
     * After this call, accessing this object can be dangerous because the 
     * process identifier may have been reused by a different process. It 
     * might still be valid, though, if the process has refused to die. 
     * 
     * \throw boost::system::system_error If the system call used to 
     *        terminate the process fails. 
     */ 
    void terminate(bool force = false) const 
    { 
#if defined(BOOST_POSIX_API) 
        if (::kill(id_, force ? SIGKILL : SIGTERM) == -1) 
            boost::throw_exception(boost::system::system_error(boost::system::error_code(errno, boost::system::get_system_category()), "boost::process::process::terminate: kill(2) failed")); 
#elif defined(BOOST_WINDOWS_API) 
        HANDLE h = ::OpenProcess(PROCESS_TERMINATE, FALSE, id_); 
        if (h == NULL) 
            boost::throw_exception(boost::system::system_error(boost::system::error_code(::GetLastError(), boost::system::get_system_category()), "boost::process::process::terminate: OpenProcess failed")); 
        if (!::TerminateProcess(h, EXIT_FAILURE)) 
        { 
            ::CloseHandle(h); 
            boost::throw_exception(boost::system::system_error(boost::system::error_code(::GetLastError(), boost::system::get_system_category()), "boost::process::process::terminate: TerminateProcess failed")); 
        } 
        if (!::CloseHandle(h)) 
            boost::throw_exception(boost::system::system_error(boost::system::error_code(::GetLastError(), boost::system::get_system_category()), "boost::process::process::terminate: CloseHandle failed")); 
#endif 
    } 

private: 
    /** 
     * The process' identifier. 
     */ 
    id_type id_; 
}; 

} 
} 

#endif 
