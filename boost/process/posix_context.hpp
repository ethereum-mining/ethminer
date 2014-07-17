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
 * \file boost/process/posix_context.hpp 
 * 
 * Includes the declaration of the posix_context class. 
 */ 

#ifndef BOOST_PROCESS_POSIX_CONTEXT_HPP 
#define BOOST_PROCESS_POSIX_CONTEXT_HPP 

#include <boost/process/context.hpp> 
#include <boost/process/stream_behavior.hpp> 
#include <map> 
#include <string> 
#include <unistd.h> 

namespace boost { 
namespace process { 

/** 
 * Holds a mapping between native file descriptors and their corresponding 
 * pipes to set up communication between the parent and the %child process. 
 */ 
typedef std::map<int, stream_behavior> behavior_map; 

template <class Path> 
class posix_basic_context : public basic_work_directory_context<Path>, public environment_context 
{ 
public: 
    /** 
     * Constructs a new POSIX-specific context. 
     * 
     * Constructs a new context. It is configured as follows: 
     * * All communcation channels with the child process are closed. 
     * * There are no channel mergings. 
     * * The initial work directory of the child processes is set to the 
     *   current working directory. 
     * * The environment variables table is empty. 
     * * The credentials are the same as those of the current process. 
     */ 
    posix_basic_context() 
        : uid(::getuid()), 
        euid(::geteuid()), 
        gid(::getgid()), 
        egid(::getegid()) 
    { 
    } 

    /** 
     * List of input streams that will be redirected. 
     */ 
    behavior_map input_behavior; 

    /** 
     * List of output streams that will be redirected. 
     */ 
    behavior_map output_behavior; 

    /** 
     * The user credentials. 
     * 
     * UID that specifies the user credentials to use to run the %child 
     * process. Defaults to the current UID. 
     */ 
    uid_t uid; 

    /** 
     * The effective user credentials. 
     * 
     * EUID that specifies the effective user credentials to use to run 
     * the %child process. Defaults to the current EUID. 
     */ 
    uid_t euid; 

    /** 
     * The group credentials. 
     * 
     * GID that specifies the group credentials to use to run the %child 
     * process. Defaults to the current GID. 
     */ 
    gid_t gid; 

    /** 
     * The effective group credentials. 
     * 
     * EGID that specifies the effective group credentials to use to run 
     * the %child process. Defaults to the current EGID. 
     */ 
    gid_t egid; 

    /** 
     * The chroot directory, if any. 
     * 
     * Specifies the directory in which the %child process is chrooted 
     * before execution. Empty if this feature is not desired. 
     */ 
    Path chroot; 
}; 

/** 
 * Default instantiation of posix_basic_context. 
 */ 
typedef posix_basic_context<std::string> posix_context; 

} 
} 

#endif 
