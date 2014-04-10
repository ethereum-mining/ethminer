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
 * \file boost/process/child.hpp 
 * 
 * Includes the declaration of the child class. 
 */ 

#ifndef BOOST_PROCESS_CHILD_HPP 
#define BOOST_PROCESS_CHILD_HPP 

#include <boost/process/config.hpp> 

#if defined(BOOST_POSIX_API) 
#  include <sys/types.h> 
#  include <sys/wait.h> 
#  include <cerrno> 
#elif defined(BOOST_WINDOWS_API) 
#  include <windows.h> 
#else 
#  error "Unsupported platform." 
#endif 

#include <boost/process/process.hpp> 
#include <boost/process/pistream.hpp> 
#include <boost/process/postream.hpp> 
#include <boost/process/status.hpp> 
#include <boost/process/detail/file_handle.hpp> 
#include <boost/system/system_error.hpp> 
#include <boost/throw_exception.hpp> 
#include <boost/shared_ptr.hpp> 
#include <boost/assert.hpp> 
#include <vector> 

namespace boost { 
namespace process { 

/** 
 * Generic implementation of the Child concept. 
 * 
 * The child class implements the Child concept in an operating system 
 * agnostic way. 
 */ 
class child : public process 
{ 
public: 
    /** 
     * Gets a reference to the child's standard input stream. 
     * 
     * Returns a reference to a postream object that represents the 
     * standard input communication channel with the child process. 
     */ 
    postream &get_stdin() const 
    { 
        BOOST_ASSERT(stdin_); 

        return *stdin_; 
    } 

    /** 
     * Gets a reference to the child's standard output stream. 
     * 
     * Returns a reference to a pistream object that represents the 
     * standard output communication channel with the child process. 
     */ 
    pistream &get_stdout() const 
    { 
        BOOST_ASSERT(stdout_); 

        return *stdout_; 
    } 

    /** 
     * Gets a reference to the child's standard error stream. 
     * 
     * Returns a reference to a pistream object that represents the 
     * standard error communication channel with the child process. 
     */ 
    pistream &get_stderr() const 
    { 
        BOOST_ASSERT(stderr_); 

        return *stderr_; 
    } 

    /** 
     * Blocks and waits for the child process to terminate. 
     * 
     * Returns a status object that represents the child process' 
     * finalization condition. The child process object ceases to be 
     * valid after this call. 
     * 
     * \remark Blocking remarks: This call blocks if the child 
     *         process has not finalized execution and waits until 
     *         it terminates. 
     */ 
    status wait() 
    { 
#if defined(BOOST_POSIX_API) 
        int s; 
        if (::waitpid(get_id(), &s, 0) == -1) 
            boost::throw_exception(boost::system::system_error(boost::system::error_code(errno, boost::system::get_system_category()), "boost::process::child::wait: waitpid(2) failed")); 
        return status(s); 
#elif defined(BOOST_WINDOWS_API) 
        ::WaitForSingleObject(process_handle_.get(), INFINITE); 
        DWORD code; 
        if (!::GetExitCodeProcess(process_handle_.get(), &code)) 
            boost::throw_exception(boost::system::system_error(boost::system::error_code(::GetLastError(), boost::system::get_system_category()), "boost::process::child::wait: GetExitCodeProcess failed")); 
        return status(code); 
#endif 
    } 

    /** 
     * Creates a new child object that represents the just spawned child 
     * process \a id. 
     * 
     * The \a fhstdin, \a fhstdout and \a fhstderr file handles represent 
     * the parent's handles used to communicate with the corresponding 
     * data streams. They needn't be valid but their availability must 
     * match the redirections configured by the launcher that spawned this 
     * process. 
     * 
     * The \a fhprocess handle represents a handle to the child process. 
     * It is only used on Windows as the implementation of wait() needs a 
     * process handle. 
     */ 
    child(id_type id, detail::file_handle fhstdin, detail::file_handle fhstdout, detail::file_handle fhstderr, detail::file_handle fhprocess = detail::file_handle()) 
        : process(id) 
#if defined(BOOST_WINDOWS_API) 
        , process_handle_(fhprocess.release(), ::CloseHandle) 
#endif 
    { 
        if (fhstdin.valid()) 
            stdin_.reset(new postream(fhstdin)); 
        if (fhstdout.valid()) 
            stdout_.reset(new pistream(fhstdout)); 
        if (fhstderr.valid()) 
            stderr_.reset(new pistream(fhstderr)); 
    } 

private: 
    /** 
     * The standard input stream attached to the child process. 
     * 
     * This postream object holds the communication channel with the 
     * child's process standard input. It is stored in a pointer because 
     * this field is only valid when the user requested to redirect this 
     * data stream. 
     */ 
    boost::shared_ptr<postream> stdin_; 

    /** 
     * The standard output stream attached to the child process. 
     * 
     * This postream object holds the communication channel with the 
     * child's process standard output. It is stored in a pointer because 
     * this field is only valid when the user requested to redirect this 
     * data stream. 
     */ 
    boost::shared_ptr<pistream> stdout_; 

    /** 
     * The standard error stream attached to the child process. 
     * 
     * This postream object holds the communication channel with the 
     * child's process standard error. It is stored in a pointer because 
     * this field is only valid when the user requested to redirect this 
     * data stream. 
     */ 
    boost::shared_ptr<pistream> stderr_; 

#if defined(BOOST_WINDOWS_API) 
    /** 
     * Process handle owned by RAII object. 
     */ 
    boost::shared_ptr<void> process_handle_; 
#endif 
}; 

/** 
 * Collection of child objects. 
 * 
 * This convenience type represents a collection of child objects backed 
 * by a vector. 
 */ 
typedef std::vector<child> children; 

} 
} 

#endif 
