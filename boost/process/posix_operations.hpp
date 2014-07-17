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
 * \file boost/process/posix_operations.hpp 
 * 
 * Provides miscellaneous free functions specific to POSIX operating 
 * systems. 
 */ 

#ifndef BOOST_PROCESS_POSIX_OPERATIONS_HPP 
#define BOOST_PROCESS_POSIX_OPERATIONS_HPP 

#include <boost/process/posix_child.hpp> 
#include <boost/process/posix_context.hpp> 
#include <boost/process/stream_behavior.hpp> 
#include <boost/process/detail/stream_info.hpp> 
#include <boost/process/detail/posix_ops.hpp> 
#include <sys/types.h> 

namespace boost { 
namespace process { 

/** 
 * Starts a new child process. 
 * 
 * Given an executable and the set of arguments passed to it, starts 
 * a new process with all the parameters configured in the context. 
 * The context can be reused afterwards to launch other different 
 * processes. 
 * 
 * \return A handle to the new child process. 
 */ 
template <class Executable, class Arguments, class Posix_Context> 
inline posix_child posix_launch(const Executable &exe, const Arguments &args, const Posix_Context &ctx) 
{ 
    detail::info_map input_info; 
    for (behavior_map::const_iterator it = ctx.input_behavior.begin(); it != ctx.input_behavior.end(); ++it) 
    { 
        if (it->second.get_type() != stream_behavior::close) 
        { 
            detail::stream_info si = detail::stream_info(it->second, false); 
            input_info.insert(detail::info_map::value_type(it->first, si)); 
        } 
    } 

    detail::info_map output_info; 
    for (behavior_map::const_iterator it = ctx.output_behavior.begin(); it != ctx.output_behavior.end(); ++it) 
    { 
        if (it->second.get_type() != stream_behavior::close) 
        { 
            detail::stream_info si = detail::stream_info(it->second, true); 
            output_info.insert(detail::info_map::value_type(it->first, si)); 
        } 
    } 

    detail::posix_setup s; 
    s.work_directory = ctx.work_directory; 
    s.uid = ctx.uid; 
    s.euid = ctx.euid; 
    s.gid = ctx.gid; 
    s.egid = ctx.egid; 
    s.chroot = ctx.chroot; 

    pid_t pid = detail::posix_start(exe, args, ctx.environment, input_info, output_info, s); 

    return posix_child(pid, input_info, output_info); 
} 

} 
} 

#endif 
