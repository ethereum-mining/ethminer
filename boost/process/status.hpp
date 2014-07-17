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
 * \file boost/process/status.hpp 
 * 
 * Includes the declaration of the status class. 
 */ 

#ifndef BOOST_PROCESS_STATUS_HPP 
#define BOOST_PROCESS_STATUS_HPP 

#include <boost/process/config.hpp> 

#if defined(BOOST_POSIX_API) 
#  include <sys/wait.h> 
#elif defined(BOOST_WINDOWS_API) 
#else 
#  error "Unsupported platform." 
#endif 

#include <boost/assert.hpp> 

namespace boost { 
namespace process { 

class child; 

/** 
 * Status returned by a finalized %child process. 
 * 
 * This class represents the %status returned by a child process after it 
 * has terminated. It only provides that information available under all 
 * supported platforms. 
 * 
 * \see posix_status 
 */ 
class status 
{ 
    friend class child; 

public: 
    /** 
     * Returns whether the process exited gracefully or not. 
     */ 
    bool exited() const 
    { 
#if defined(BOOST_POSIX_API) 
        return WIFEXITED(flags_); 
#elif defined(BOOST_WINDOWS_API) 
        return true; 
#endif 
    } 

    /** 
     * If exited, returns the exit code. 
     * 
     * If the process exited, returns the exit code it returned. 
     * 
     * \pre exited() is true. 
     */ 
    int exit_status() const 
    { 
        BOOST_ASSERT(exited()); 
#if defined(BOOST_POSIX_API) 
        return WEXITSTATUS(flags_); 
#elif defined(BOOST_WINDOWS_API) 
        return flags_; 
#endif 
    } 

protected: 
    /** 
     * Creates a status object based on exit information. 
     * 
     * Creates a new status object representing the exit status of a 
     * child process. 
     * 
     * \param flags In a POSIX system this parameter contains the 
     *        flags returned by the ::waitpid() call. In a 
     *        Windows system it contains the exit code only. 
     */ 
    status(int flags) 
        : flags_(flags) 
    { 
    } 

    /** 
     * OS-specific codification of exit status. 
     */ 
    int flags_; 
}; 

} 
} 

#endif 
