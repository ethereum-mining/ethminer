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
 * \file boost/process/posix_status.hpp 
 * 
 * Includes the declaration of the posix_status class. 
 */ 

#ifndef BOOST_PROCESS_POSIX_STATUS_HPP 
#define BOOST_PROCESS_POSIX_STATUS_HPP 

#include <boost/process/config.hpp> 

#if defined(BOOST_POSIX_API) 
#  include <sys/wait.h> 
#elif defined(BOOST_WINDOWS_API) 
#else 
#  error "Unsupported platform." 
#endif 

#include <boost/process/status.hpp> 
#include <boost/assert.hpp> 

namespace boost { 
namespace process { 

/** 
 * Status returned by a finalized %child process on a POSIX system. 
 * 
 * This class represents the %status returned by a child process after it 
 * has terminated. It contains some methods not available in the status 
 * class that provide information only available in POSIX systems. 
 */ 
class posix_status : public status 
{ 
public: 
    /** 
     * Creates a posix_status object from an existing status object. 
     * 
     * Creates a new status object representing the exit status of a 
     * child process. The construction is done based on an existing 
     * status object which already contains all the available 
     * information: this class only provides controlled access to it. 
     */ 
    posix_status(const status &s) 
        : status(s) 
    { 
    } 

    /** 
     * Returns whether the process exited due to an external 
     * signal. 
     */ 
    bool signaled() const 
    { 
        return WIFSIGNALED(flags_); 
    } 

    /** 
     * If signaled, returns the terminating signal code. 
     * 
     * If the process was signaled, returns the terminating signal code. 
     * 
     * \pre signaled() is true. 
     */ 
    int term_signal() const 
    { 
        BOOST_ASSERT(signaled()); 

        return WTERMSIG(flags_); 
    } 

    /** 
     * If signaled, returns whether the process dumped core. 
     * 
     * If the process was signaled, returns whether the process 
     * produced a core dump. 
     * 
     * \pre signaled() is true. 
     */ 
    bool dumped_core() const 
    { 
        BOOST_ASSERT(signaled()); 

#ifdef WCOREDUMP 
        return WCOREDUMP(flags_); 
#else 
        return false; 
#endif 
    } 

    /** 
     * Returns whether the process was stopped by an external 
     * signal. 
     */ 
    bool stopped() const 
    { 
        return WIFSTOPPED(flags_); 
    } 

    /** 
     * If stopped, returns the stop signal code. 
     * 
     * If the process was stopped, returns the stop signal code. 
     * 
     * \pre stopped() is true. 
     */ 
    int stop_signal() const 
    { 
        BOOST_ASSERT(stopped()); 

        return WSTOPSIG(flags_); 
    } 
}; 

} 
} 

#endif 
