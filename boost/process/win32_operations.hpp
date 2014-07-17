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
 * \file boost/process/win32_operations.hpp 
 * 
 * Provides miscellaneous free functions specific to Windows operating 
 * systems. 
 */ 

#ifndef BOOST_PROCESS_WIN32_OPERATIONS_HPP 
#define BOOST_PROCESS_WIN32_OPERATIONS_HPP 

#include <boost/process/win32_child.hpp> 
#include <boost/process/detail/file_handle.hpp> 
#include <boost/process/detail/stream_info.hpp> 
#include <boost/process/detail/win32_ops.hpp> 
#include <windows.h> 

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
template <class Executable, class Arguments, class Win32_Context> 
inline win32_child win32_launch(const Executable &exe, const Arguments &args, const Win32_Context &ctx) 
{ 
    detail::file_handle fhstdin, fhstdout, fhstderr; 

    detail::stream_info behin = detail::stream_info(ctx.stdin_behavior, false); 
    if (behin.type_ == detail::stream_info::use_pipe) 
        fhstdin = behin.pipe_->wend(); 
    detail::stream_info behout = detail::stream_info(ctx.stdout_behavior, true); 
    if (behout.type_ == detail::stream_info::use_pipe) 
        fhstdout = behout.pipe_->rend(); 
    detail::stream_info beherr = detail::stream_info(ctx.stderr_behavior, true); 
    if (beherr.type_ == detail::stream_info::use_pipe) 
        fhstderr = beherr.pipe_->rend(); 

    detail::win32_setup s; 
    s.work_directory = ctx.work_directory; 

    STARTUPINFOA si; 
    if (!ctx.startupinfo) 
    { 
        ::ZeroMemory(&si, sizeof(si)); 
        si.cb = sizeof(si); 
        s.startupinfo = &si; 
    } 
    else 
        s.startupinfo = ctx.startupinfo; 

    PROCESS_INFORMATION pi = detail::win32_start(exe, args, ctx.environment, behin, behout, beherr, s); 

    return win32_child(pi, fhstdin, fhstdout, fhstderr); 
} 

} 
} 

#endif 
