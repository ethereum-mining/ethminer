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
 * \file boost/process/detail/win32_ops.hpp 
 * 
 * Provides some convenience functions to start processes under 
 * Windows-compatible operating systems. 
 */ 

#ifndef BOOST_PROCESS_DETAIL_WIN32_OPS_HPP 
#define BOOST_PROCESS_DETAIL_WIN32_OPS_HPP 

#include <boost/process/environment.hpp> 
#include <boost/process/detail/file_handle.hpp> 
#include <boost/process/detail/pipe.hpp> 
#include <boost/process/detail/stream_info.hpp> 
#include <boost/scoped_ptr.hpp> 
#include <boost/shared_array.hpp> 
#include <boost/scoped_array.hpp> 
#include <boost/assert.hpp> 
#include <boost/system/system_error.hpp> 
#include <boost/throw_exception.hpp> 
#include <vector> 
#include <string> 
#include <cstddef> 
#include <string.h> 
#include <windows.h> 

namespace boost { 
namespace process { 
namespace detail { 

/** 
 * Converts the command line to a plain string. Converts the command line's 
 * list of arguments to the format expected by the \a lpCommandLine parameter 
 * in the CreateProcess() system call. 
 * 
 * This operation is only available on Windows systems. 
 * 
 * \return A dynamically allocated string holding the command line 
 *         to be passed to the executable. It is returned in a 
 *         shared_array object to ensure its release at some point. 
 */ 
template <class Arguments> 
inline boost::shared_array<char> collection_to_win32_cmdline(const Arguments &args) 
{ 
    typedef std::vector<std::string> arguments_t; 
    arguments_t args2; 
    typename Arguments::size_type i = 0; 
    std::size_t size = 0; 
    for (typename Arguments::const_iterator it = args.begin(); it != args.end(); ++it) 
    { 
        std::string arg = *it; 

        std::string::size_type pos = 0; 
        while ( (pos = arg.find('"', pos)) != std::string::npos) 
        { 
            arg.replace(pos, 1, "\\\""); 
            pos += 2; 
        } 

        if (arg.find(' ') != std::string::npos) 
            arg = '\"' + arg + '\"'; 

        if (i++ != args.size() - 1) 
            arg += ' '; 

        args2.push_back(arg); 
        size += arg.size() + 1; 
    } 

    boost::shared_array<char> cmdline(new char[size]); 
    cmdline.get()[0] = '\0'; 
    for (arguments_t::size_type i = 0; i < args.size(); ++i) 
#if defined(__CYGWIN__) 
        ::strncat(cmdline.get(), args2[i].c_str(), args2[i].size()); 
#else 
        ::strcat_s(cmdline.get(), size, args2[i].c_str()); 
#endif 

    return cmdline; 
} 

/** 
 * Converts an environment to a string used by CreateProcess(). 
 * 
 * Converts the environment's contents to the format used by the 
 * CreateProcess() system call. The returned char* string is 
 * allocated in dynamic memory; the caller must free it when not 
 * used any more. This is enforced by the use of a shared pointer. 
 * 
 * \return A dynamically allocated char* string that represents 
 *         the environment's content. This string is of the form 
 *         var1=value1\\0var2=value2\\0\\0. 
 */ 
inline boost::shared_array<char> environment_to_win32_strings(const environment &env) 
{ 
    boost::shared_array<char> envp; 

    if (env.empty()) 
    { 
        envp.reset(new char[2]); 
        ::ZeroMemory(envp.get(), 2); 
    } 
    else 
    { 
        std::string s; 
        for (environment::const_iterator it = env.begin(); it != env.end(); ++it) 
        { 
            s += it->first + "=" + it->second; 
            s.push_back(0); 
        } 

        envp.reset(new char[s.size() + 1]); 
#if defined(__CYGWIN__) 
        ::memcpy(envp.get(), s.c_str(), s.size() + 1); 
#else 
        ::memcpy_s(envp.get(), s.size() + 1, s.c_str(), s.size() + 1); 
#endif 
    } 

    return envp; 
} 

/** 
 * Helper class to configure a Win32 %child. 
 * 
 * This helper class is used to hold all the attributes that configure a 
 * new Win32 %child process. 
 * 
 * All its fields are public for simplicity. It is only intended for 
 * internal use and it is heavily coupled with the Context 
 * implementations. 
 */ 
struct win32_setup 
{ 
    /** 
     * The work directory. 
     * 
     * This string specifies the directory in which the %child process 
     * starts execution. It cannot be empty. 
     */ 
    std::string work_directory; 

    /** 
     * The process startup properties. 
     * 
     * This Win32-specific object holds a list of properties that describe 
     * how the new process should be started. The \a STARTF_USESTDHANDLES 
     * flag should not be set in it because it is automatically configured 
     * by win32_start(). 
     */ 
    STARTUPINFOA *startupinfo; 
}; 

/** 
 * Starts a new child process in a Win32 operating system. 
 * 
 * This helper functions is provided to simplify the Context's task when 
 * it comes to starting up a new process in a Win32 operating system. 
 * 
 * \param exe The executable to spawn the child process. 
 * \param args The arguments for the executable. 
 * \param env The environment variables that the new child process 
 *        receives. 
 * \param infoin Information that describes stdin's behavior. 
 * \param infoout Information that describes stdout's behavior. 
 * \param infoerr Information that describes stderr's behavior. 
 * \param setup A helper object holding extra child information. 
 * \return The new process' information as returned by the CreateProcess() 
 *         system call. The caller is responsible of creating an 
 *         appropriate Child representation for it. 
 * \pre \a setup.startupinfo_ cannot have the \a STARTF_USESTDHANDLES set 
 *      in the \a dwFlags field. 
 */ 
template <class Executable, class Arguments> 
inline PROCESS_INFORMATION win32_start(const Executable &exe, const Arguments &args, const environment &env, stream_info &infoin, stream_info &infoout, stream_info &infoerr, const win32_setup &setup) 
{ 
    file_handle chin, chout, cherr; 

    BOOST_ASSERT(setup.startupinfo->cb >= sizeof(STARTUPINFOA)); 
    BOOST_ASSERT(!(setup.startupinfo->dwFlags & STARTF_USESTDHANDLES)); 

    boost::scoped_ptr<STARTUPINFOA> si(new STARTUPINFOA); 
    ::CopyMemory(si.get(), setup.startupinfo, sizeof(*setup.startupinfo)); 
    si->dwFlags |= STARTF_USESTDHANDLES; 

    switch (infoin.type_) 
    { 
    case stream_info::close: 
        { 
            break; 
        } 
    case stream_info::inherit: 
        { 
            chin = file_handle::win32_std(STD_INPUT_HANDLE, true); 
            break; 
        } 
    case stream_info::use_file: 
        { 
            HANDLE h = ::CreateFileA(infoin.file_.c_str(), GENERIC_READ, 0, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL); 
            if (h == INVALID_HANDLE_VALUE) 
                boost::throw_exception(boost::system::system_error(boost::system::error_code(::GetLastError(), boost::system::get_system_category()), "boost::process::detail::win32_start: CreateFile failed")); 
            chin = file_handle(h); 
            break; 
        } 
    case stream_info::use_handle: 
        { 
            chin = infoin.handle_; 
            chin.win32_set_inheritable(true); 
            break; 
        } 
    case stream_info::use_pipe: 
        { 
            infoin.pipe_->rend().win32_set_inheritable(true); 
            chin = infoin.pipe_->rend(); 
            break; 
        } 
    default: 
        { 
            BOOST_ASSERT(false); 
            break; 
        } 
    } 

    si->hStdInput = chin.valid() ? chin.get() : INVALID_HANDLE_VALUE; 

    switch (infoout.type_) 
    { 
    case stream_info::close: 
        { 
            break; 
        } 
    case stream_info::inherit: 
        { 
            chout = file_handle::win32_std(STD_OUTPUT_HANDLE, true); 
            break; 
        } 
    case stream_info::use_file: 
        { 
            HANDLE h = ::CreateFileA(infoout.file_.c_str(), GENERIC_WRITE, 0, NULL, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL); 
            if (h == INVALID_HANDLE_VALUE) 
                boost::throw_exception(boost::system::system_error(boost::system::error_code(::GetLastError(), boost::system::get_system_category()), "boost::process::detail::win32_start: CreateFile failed")); 
            chout = file_handle(h); 
            break; 
        } 
    case stream_info::use_handle: 
        { 
            chout = infoout.handle_; 
            chout.win32_set_inheritable(true); 
            break; 
        } 
    case stream_info::use_pipe: 
        { 
            infoout.pipe_->wend().win32_set_inheritable(true); 
            chout = infoout.pipe_->wend(); 
            break; 
        } 
    default: 
        { 
            BOOST_ASSERT(false); 
            break; 
        } 
    } 

    si->hStdOutput = chout.valid() ? chout.get() : INVALID_HANDLE_VALUE; 

    switch (infoerr.type_) 
    { 
    case stream_info::close: 
        { 
            break; 
        } 
    case stream_info::inherit: 
        { 
            cherr = file_handle::win32_std(STD_ERROR_HANDLE, true); 
            break; 
        } 
    case stream_info::redirect: 
        { 
            BOOST_ASSERT(infoerr.desc_to_ == 1); 
            BOOST_ASSERT(chout.valid()); 
            cherr = file_handle::win32_dup(chout.get(), true); 
            break; 
        } 
    case stream_info::use_file: 
        { 
            HANDLE h = ::CreateFileA(infoerr.file_.c_str(), GENERIC_WRITE, 0, NULL, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL); 
            if (h == INVALID_HANDLE_VALUE) 
                boost::throw_exception(boost::system::system_error(boost::system::error_code(::GetLastError(), boost::system::get_system_category()), "boost::process::detail::win32_start: CreateFile failed")); 
            cherr = file_handle(h); 
            break; 
        } 
    case stream_info::use_handle: 
        { 
            cherr = infoerr.handle_; 
            cherr.win32_set_inheritable(true); 
            break; 
        } 
    case stream_info::use_pipe: 
        { 
            infoerr.pipe_->wend().win32_set_inheritable(true); 
            cherr = infoerr.pipe_->wend(); 
            break; 
        } 
    default: 
        { 
            BOOST_ASSERT(false); 
            break; 
        } 
    } 

    si->hStdError = cherr.valid() ? cherr.get() : INVALID_HANDLE_VALUE; 

    PROCESS_INFORMATION pi; 
    ::ZeroMemory(&pi, sizeof(pi)); 

    boost::shared_array<char> cmdline = collection_to_win32_cmdline(args); 

    boost::scoped_array<char> executable(new char[exe.size() + 1]); 
#if defined(__CYGWIN__) 
    ::strcpy(executable.get(), exe.c_str()); 
#else 
    ::strcpy_s(executable.get(), exe.size() + 1, exe.c_str()); 
#endif 

    boost::scoped_array<char> workdir(new char[setup.work_directory.size() + 1]); 
#if defined(__CYGWIN__) 
    ::strcpy(workdir.get(), setup.work_directory.c_str()); 
#else 
    ::strcpy_s(workdir.get(), setup.work_directory.size() + 1, setup.work_directory.c_str()); 
#endif 

    boost::shared_array<char> envstrs = environment_to_win32_strings(env); 

    if (!::CreateProcessA(executable.get(), cmdline.get(), NULL, NULL, TRUE, 0, envstrs.get(), workdir.get(), si.get(), &pi)) 
        boost::throw_exception(boost::system::system_error(boost::system::error_code(::GetLastError(), boost::system::get_system_category()), "boost::process::detail::win32_start: CreateProcess failed")); 

    return pi; 
} 

} 
} 
} 

#endif 
