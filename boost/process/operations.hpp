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
 * \file boost/process/operations.hpp 
 * 
 * Provides miscellaneous free functions. 
 */ 

#ifndef BOOST_PROCESS_OPERATIONS_HPP 
#define BOOST_PROCESS_OPERATIONS_HPP 

#include <boost/process/config.hpp> 

#if defined(BOOST_POSIX_API) 
#  include <boost/process/detail/posix_ops.hpp> 
#  include <stdlib.h> 
#  include <unistd.h> 
#  if defined(__CYGWIN__) 
#    include <boost/scoped_array.hpp> 
#    include <sys/cygwin.h> 
#  endif 
#elif defined(BOOST_WINDOWS_API) 
#  include <boost/process/detail/win32_ops.hpp> 
#  include <boost/algorithm/string/predicate.hpp> 
#  include <windows.h> 
#else 
#  error "Unsupported platform." 
#endif 

#include <boost/process/child.hpp> 
#include <boost/process/stream_behavior.hpp> 
#include <boost/process/status.hpp> 
#include <boost/process/detail/file_handle.hpp> 
#include <boost/process/detail/pipe.hpp> 
#include <boost/process/detail/stream_info.hpp> 
#include <boost/filesystem/path.hpp> 
#include <boost/algorithm/string/predicate.hpp> 
#include <boost/system/system_error.hpp> 
#include <boost/throw_exception.hpp> 
#include <boost/scoped_array.hpp> 
#include <boost/assert.hpp> 
#include <string> 
#include <vector> 
#include <stdexcept> 
#include <cstddef> 

namespace boost { 
namespace process { 

/** 
 * Locates the executable program \a file in all the directory components 
 * specified in \a path. If \a path is empty, the value of the PATH 
 * environment variable is used. 
 * 
 * The path variable is interpreted following the same conventions used 
 * to parse the PATH environment variable in the underlying platform. 
 * 
 * \throw boost::filesystem::filesystem_error If the file cannot be found 
 *        in the path. 
 */ 
inline std::string find_executable_in_path(const std::string &file, std::string path = "") 
{ 
#if defined(BOOST_POSIX_API) 
    BOOST_ASSERT(file.find('/') == std::string::npos); 
#elif defined(BOOST_WINDOWS_API) 
    BOOST_ASSERT(file.find_first_of("\\/") == std::string::npos); 
#endif 

    std::string result; 

#if defined(BOOST_POSIX_API) 
    if (path.empty()) 
    { 
        const char *envpath = ::getenv("PATH"); 
//        if (!envpath)
 //           boost::throw_exception(boost::filesystem::filesystem_error("boost::process::find_executable_in_path: retrieving PATH failed", file, boost::system::errc::make_error_code(boost::system::errc::no_such_file_or_directory)));

        path = envpath; 
    } 
    BOOST_ASSERT(!path.empty()); 

#if defined(__CYGWIN__) 
    if (!::cygwin_posix_path_list_p(path.c_str())) 
    { 
        int size = ::cygwin_win32_to_posix_path_list_buf_size(path.c_str()); 
        boost::scoped_array<char> cygpath(new char[size]); 
        ::cygwin_win32_to_posix_path_list(path.c_str(), cygpath.get()); 
        path = cygpath.get(); 
    } 
#endif 

    std::string::size_type pos1 = 0, pos2; 
    do 
    { 
        pos2 = path.find(':', pos1); 
        std::string dir = (pos2 != std::string::npos) ? path.substr(pos1, pos2 - pos1) : path.substr(pos1); 
        std::string f = dir + (boost::algorithm::ends_with(dir, "/") ? "" : "/") + file; 
        if (!::access(f.c_str(), X_OK)) 
            result = f; 
        pos1 = pos2 + 1; 
    } while (pos2 != std::string::npos && result.empty()); 
#elif defined(BOOST_WINDOWS_API) 
    const char *exts[] = { "", ".exe", ".com", ".bat", NULL }; 
    const char **ext = exts; 
    while (*ext) 
    { 
        char buf[MAX_PATH]; 
        char *dummy; 
        DWORD size = ::SearchPathA(path.empty() ? NULL : path.c_str(), file.c_str(), *ext, MAX_PATH, buf, &dummy); 
        BOOST_ASSERT(size < MAX_PATH); 
        if (size > 0) 
        { 
            result = buf; 
            break; 
        } 
        ++ext; 
    } 
#endif 

 //   if (result.empty())
//        boost::throw_exception(boost::filesystem::filesystem_error("boost::process::find_executable_in_path: file not found", file, boost::system::errc::make_error_code(boost::system::errc::no_such_file_or_directory)));

    return result; 
} 

/** 
 * Extracts the program name from a given executable. 
 * 
 * \return The program name. On Windows the program name 
 *         is returned without a file extension. 
 */ 
inline std::string executable_to_progname(const std::string &exe) 
{ 
    std::string::size_type begin = 0; 
    std::string::size_type end = std::string::npos; 

#if defined(BOOST_POSIX_API) 
    std::string::size_type slash = exe.rfind('/'); 
#elif defined(BOOST_WINDOWS_API) 
    std::string::size_type slash = exe.find_last_of("/\\"); 
#endif 
    if (slash != std::string::npos) 
        begin = slash + 1; 

#if defined(BOOST_WINDOWS_API) 
    if (exe.size() > 4 && 
        (boost::algorithm::iends_with(exe, ".exe") || boost::algorithm::iends_with(exe, ".com") || boost::algorithm::iends_with(exe, ".bat"))) 
        end = exe.size() - 4; 
#endif 

    return exe.substr(begin, end - begin); 
} 

/** 
 * Starts a new child process. 
 * 
 * Launches a new process based on the binary image specified by the 
 * executable, the set of arguments to be passed to it and several 
 * parameters that describe the execution context. 
 * 
 * \remark Blocking remarks: This function may block if the device 
 * holding the executable blocks when loading the image. This might 
 * happen if, e.g., the binary is being loaded from a network share. 
 * 
 * \return A handle to the new child process. 
 */ 
template <class Executable, class Arguments, class Context> 
inline child launch(const Executable &exe, const Arguments &args, const Context &ctx) 
{ 
    detail::file_handle fhstdin, fhstdout, fhstderr; 

    BOOST_ASSERT(!args.empty()); 
    BOOST_ASSERT(!ctx.work_directory.empty()); 

#if defined(BOOST_POSIX_API) 
    detail::info_map infoin, infoout; 

    if (ctx.stdin_behavior.get_type() != stream_behavior::close) 
    { 
        detail::stream_info si = detail::stream_info(ctx.stdin_behavior, false); 
        infoin.insert(detail::info_map::value_type(STDIN_FILENO, si)); 
    } 

    if (ctx.stdout_behavior.get_type() != stream_behavior::close) 
    { 
        detail::stream_info si = detail::stream_info(ctx.stdout_behavior, true); 
        infoout.insert(detail::info_map::value_type(STDOUT_FILENO, si)); 
    } 

    if (ctx.stderr_behavior.get_type() != stream_behavior::close) 
    { 
        detail::stream_info si = detail::stream_info(ctx.stderr_behavior, true); 
        infoout.insert(detail::info_map::value_type(STDERR_FILENO, si)); 
    } 

    detail::posix_setup s; 
    s.work_directory = ctx.work_directory; 

    child::id_type pid = detail::posix_start(exe, args, ctx.environment, infoin, infoout, s); 

    if (ctx.stdin_behavior.get_type() == stream_behavior::capture) 
    { 
        fhstdin = detail::posix_info_locate_pipe(infoin, STDIN_FILENO, false); 
        BOOST_ASSERT(fhstdin.valid()); 
    } 

    if (ctx.stdout_behavior.get_type() == stream_behavior::capture) 
    { 
        fhstdout = detail::posix_info_locate_pipe(infoout, STDOUT_FILENO, true); 
        BOOST_ASSERT(fhstdout.valid()); 
    } 

    if (ctx.stderr_behavior.get_type() == stream_behavior::capture) 
    { 
        fhstderr = detail::posix_info_locate_pipe(infoout, STDERR_FILENO, true); 
        BOOST_ASSERT(fhstderr.valid()); 
    } 

    return child(pid, fhstdin, fhstdout, fhstderr); 
#elif defined(BOOST_WINDOWS_API) 
    detail::stream_info behin = detail::stream_info(ctx.stdin_behavior, false); 
    if (behin.type_ == detail::stream_info::use_pipe) 
        fhstdin = behin.pipe_->wend(); 
    detail::stream_info behout = detail::stream_info(ctx.stdout_behavior, true); 
    if (behout.type_ == detail::stream_info::use_pipe) 
        fhstdout = behout.pipe_->rend(); 
    detail::stream_info beherr = detail::stream_info(ctx.stderr_behavior, true); 
    if (beherr.type_ == detail::stream_info::use_pipe) 
        fhstderr = beherr.pipe_->rend(); 

    STARTUPINFOA si; 
    ::ZeroMemory(&si, sizeof(si)); 
    si.cb = sizeof(si); 

    detail::win32_setup s; 
    s.work_directory = ctx.work_directory; 
    s.startupinfo = &si; 

    PROCESS_INFORMATION pi = detail::win32_start(exe, args, ctx.environment, behin, behout, beherr, s); 

    if (!::CloseHandle(pi.hThread)) 
        boost::throw_exception(boost::system::system_error(boost::system::error_code(::GetLastError(), boost::system::get_system_category()), "boost::process::launch: CloseHandle failed")); 

    return child(pi.dwProcessId, fhstdin, fhstdout, fhstderr, detail::file_handle(pi.hProcess)); 
#endif 
} 

/** 
 * Launches a shell-based command. 
 * 
 * Executes the given command through the default system shell. The 
 * command is subject to pattern expansion, redirection and pipelining. 
 * The shell is launched as described by the parameters in the execution 
 * context. 
 * 
 * This function behaves similarly to the system(3) system call. In a 
 * POSIX system, the command is fed to /bin/sh whereas under a Windows 
 * system, it is fed to cmd.exe. It is difficult to write portable 
 * commands as the first parameter, but this function comes in handy in 
 * multiple situations. 
 */ 
template <class Context> 
inline child launch_shell(const std::string &command, const Context &ctx) 
{ 
    std::string exe; 
    std::vector<std::string> args; 

#if defined(BOOST_POSIX_API) 
    exe = "/bin/sh"; 
    args.push_back("sh"); 
    args.push_back("-c"); 
    args.push_back(command); 
#elif defined(BOOST_WINDOWS_API) 
    char sysdir[MAX_PATH]; 
    UINT size = ::GetSystemDirectoryA(sysdir, sizeof(sysdir)); 
    if (!size) 
        boost::throw_exception(boost::system::system_error(boost::system::error_code(::GetLastError(), boost::system::get_system_category()), "boost::process::launch_shell: GetWindowsDirectory failed")); 
    BOOST_ASSERT(size < MAX_PATH); 

    exe = std::string(sysdir) + (sysdir[size - 1] != '\\' ? "\\cmd.exe" : "cmd.exe"); 
    args.push_back("cmd"); 
    args.push_back("/c"); 
    args.push_back(command); 
#endif 

    return launch(exe, args, ctx); 
} 

/** 
 * Launches a pipelined set of child processes. 
 * 
 * Given a collection of pipeline_entry objects describing how to launch 
 * a set of child processes, spawns them all and connects their inputs and 
 * outputs in a way that permits pipelined communication. 
 * 
 * \pre Let 1..N be the processes in the collection: the input behavior of 
 *      the 2..N processes must be set to close_stream(). 
 * \pre Let 1..N be the processes in the collection: the output behavior of 
 *      the 1..N-1 processes must be set to close_stream(). 
 * \remark Blocking remarks: This function may block if the device holding 
 *         the executable of one of the entries blocks when loading the 
 *         image. This might happen if, e.g., the binary is being loaded 
 *         from a network share. 
 * \return A set of Child objects that represent all the processes spawned 
 *         by this call. You should use wait_children() to wait for their 
 *         termination. 
 */ 
template <class Entries> 
inline children launch_pipeline(const Entries &entries) 
{ 
    BOOST_ASSERT(entries.size() >= 2); 

    children cs; 
    detail::file_handle fhinvalid; 

    boost::scoped_array<detail::pipe> pipes(new detail::pipe[entries.size() - 1]); 

#if defined(BOOST_POSIX_API) 
    { 
        typename Entries::size_type i = 0; 
        const typename Entries::value_type::context_type &ctx = entries[i].context; 

        detail::info_map infoin, infoout; 

        if (ctx.stdin_behavior.get_type() != stream_behavior::close) 
        { 
            detail::stream_info si = detail::stream_info(ctx.stdin_behavior, false); 
            infoin.insert(detail::info_map::value_type(STDIN_FILENO, si)); 
        } 

        BOOST_ASSERT(ctx.stdout_behavior.get_type() == stream_behavior::close); 
        detail::stream_info si2(close_stream(), true); 
        si2.type_ = detail::stream_info::use_handle; 
        si2.handle_ = pipes[i].wend().release(); 
        infoout.insert(detail::info_map::value_type(STDOUT_FILENO, si2)); 

        if (ctx.stderr_behavior.get_type() != stream_behavior::close) 
        { 
            detail::stream_info si = detail::stream_info(ctx.stderr_behavior, true); 
            infoout.insert(detail::info_map::value_type(STDERR_FILENO, si)); 
        } 

        detail::posix_setup s; 
        s.work_directory = ctx.work_directory; 

        pid_t pid = detail::posix_start(entries[i].executable, entries[i].arguments, ctx.environment, infoin, infoout, s); 

        detail::file_handle fhstdin; 

        if (ctx.stdin_behavior.get_type() == stream_behavior::capture) 
        { 
            fhstdin = detail::posix_info_locate_pipe(infoin, STDIN_FILENO, false); 
            BOOST_ASSERT(fhstdin.valid()); 
        } 

        cs.push_back(child(pid, fhstdin, fhinvalid, fhinvalid)); 
    } 

    for (typename Entries::size_type i = 1; i < entries.size() - 1; ++i) 
    { 
        const typename Entries::value_type::context_type &ctx = entries[i].context; 
        detail::info_map infoin, infoout; 

        BOOST_ASSERT(ctx.stdin_behavior.get_type() == stream_behavior::close); 
        detail::stream_info si1(close_stream(), false); 
        si1.type_ = detail::stream_info::use_handle; 
        si1.handle_ = pipes[i - 1].rend().release(); 
        infoin.insert(detail::info_map::value_type(STDIN_FILENO, si1)); 

        BOOST_ASSERT(ctx.stdout_behavior.get_type() == stream_behavior::close); 
        detail::stream_info si2(close_stream(), true); 
        si2.type_ = detail::stream_info::use_handle; 
        si2.handle_ = pipes[i].wend().release(); 
        infoout.insert(detail::info_map::value_type(STDOUT_FILENO, si2)); 

        if (ctx.stderr_behavior.get_type() != stream_behavior::close) 
        { 
            detail::stream_info si = detail::stream_info(ctx.stderr_behavior, true); 
            infoout.insert(detail::info_map::value_type(STDERR_FILENO, si)); 
        } 

        detail::posix_setup s; 
        s.work_directory = ctx.work_directory; 

        pid_t pid = detail::posix_start(entries[i].executable, entries[i].arguments, ctx.environment, infoin, infoout, s); 

        cs.push_back(child(pid, fhinvalid, fhinvalid, fhinvalid)); 
    } 

    { 
        typename Entries::size_type i = entries.size() - 1; 
        const typename Entries::value_type::context_type &ctx = entries[i].context; 

        detail::info_map infoin, infoout; 

        BOOST_ASSERT(ctx.stdin_behavior.get_type() == stream_behavior::close); 
        detail::stream_info si1(close_stream(), false); 
        si1.type_ = detail::stream_info::use_handle; 
        si1.handle_ = pipes[i - 1].rend().release(); 
        infoin.insert(detail::info_map::value_type(STDIN_FILENO, si1)); 

        if (ctx.stdout_behavior.get_type() != stream_behavior::close) 
        { 
            detail::stream_info si = detail::stream_info(ctx.stdout_behavior, true); 
            infoout.insert(detail::info_map::value_type(STDOUT_FILENO, si)); 
        } 

        if (ctx.stderr_behavior.get_type() != stream_behavior::close) 
        { 
            detail::stream_info si = detail::stream_info(ctx.stderr_behavior, true); 
            infoout.insert(detail::info_map::value_type(STDERR_FILENO, si)); 
        } 

        detail::posix_setup s; 
        s.work_directory = ctx.work_directory; 

        pid_t pid = detail::posix_start(entries[i].executable, entries[i].arguments, ctx.environment, infoin, infoout, s); 

        detail::file_handle fhstdout, fhstderr; 

        if (ctx.stdout_behavior.get_type() == stream_behavior::capture) 
        { 
            fhstdout = detail::posix_info_locate_pipe(infoout, STDOUT_FILENO, true); 
            BOOST_ASSERT(fhstdout.valid()); 
        } 

        if (ctx.stderr_behavior.get_type() == stream_behavior::capture) 
        { 
            fhstderr = detail::posix_info_locate_pipe(infoout, STDERR_FILENO, true); 
            BOOST_ASSERT(fhstderr.valid()); 
        } 

        cs.push_back(child(pid, fhinvalid, fhstdout, fhstderr)); 
    } 
#elif defined(BOOST_WINDOWS_API) 
    STARTUPINFOA si; 
    detail::win32_setup s; 
    s.startupinfo = &si; 

    { 
        typename Entries::size_type i = 0; 
        const typename Entries::value_type::context_type &ctx = entries[i].context; 

        detail::stream_info sii = detail::stream_info(ctx.stdin_behavior, false); 
        detail::file_handle fhstdin; 
        if (sii.type_ == detail::stream_info::use_pipe) 
            fhstdin = sii.pipe_->wend(); 

        BOOST_ASSERT(ctx.stdout_behavior.get_type() == stream_behavior::close); 
        detail::stream_info sio(close_stream(), true); 
        sio.type_ = detail::stream_info::use_handle; 
        sio.handle_ = pipes[i].wend().release(); 

        detail::stream_info sie(ctx.stderr_behavior, true); 

        s.work_directory = ctx.work_directory; 

        ::ZeroMemory(&si, sizeof(si)); 
        si.cb = sizeof(si); 
        PROCESS_INFORMATION pi = detail::win32_start(entries[i].executable, entries[i].arguments, ctx.environment, sii, sio, sie, s); 

        if (!::CloseHandle(pi.hThread)) 
            boost::throw_exception(boost::system::system_error(boost::system::error_code(::GetLastError(), boost::system::get_system_category()), "boost::process::launch_pipeline: CloseHandle failed")); 

        cs.push_back(child(pi.dwProcessId, fhstdin, fhinvalid, fhinvalid, detail::file_handle(pi.hProcess))); 
    } 

    for (typename Entries::size_type i = 1; i < entries.size() - 1; ++i) 
    { 
        const typename Entries::value_type::context_type &ctx = entries[i].context; 

        BOOST_ASSERT(ctx.stdin_behavior.get_type() == stream_behavior::close); 
        detail::stream_info sii(close_stream(), false); 
        sii.type_ = detail::stream_info::use_handle; 
        sii.handle_ = pipes[i - 1].rend().release(); 

        detail::stream_info sio(close_stream(), true); 
        sio.type_ = detail::stream_info::use_handle; 
        sio.handle_ = pipes[i].wend().release(); 

        detail::stream_info sie(ctx.stderr_behavior, true); 

        s.work_directory = ctx.work_directory; 

        ::ZeroMemory(&si, sizeof(si)); 
        si.cb = sizeof(si); 
        PROCESS_INFORMATION pi = detail::win32_start(entries[i].executable, entries[i].arguments, ctx.environment, sii, sio, sie, s); 

        if (!::CloseHandle(pi.hThread)) 
            boost::throw_exception(boost::system::system_error(boost::system::error_code(::GetLastError(), boost::system::get_system_category()), "boost::process::launch_pipeline: CloseHandle failed")); 

        cs.push_back(child(pi.dwProcessId, fhinvalid, fhinvalid, fhinvalid, detail::file_handle(pi.hProcess))); 
    } 

    { 
        typename Entries::size_type i = entries.size() - 1; 
        const typename Entries::value_type::context_type &ctx = entries[i].context; 

        BOOST_ASSERT(ctx.stdin_behavior.get_type() == stream_behavior::close); 
        detail::stream_info sii(close_stream(), false); 
        sii.type_ = detail::stream_info::use_handle; 
        sii.handle_ = pipes[i - 1].rend().release(); 

        detail::file_handle fhstdout, fhstderr; 

        detail::stream_info sio(ctx.stdout_behavior, true); 
        if (sio.type_ == detail::stream_info::use_pipe) 
            fhstdout = sio.pipe_->rend(); 
        detail::stream_info sie(ctx.stderr_behavior, true); 
        if (sie.type_ == detail::stream_info::use_pipe) 
            fhstderr = sie.pipe_->rend(); 

        s.work_directory = ctx.work_directory; 

        ::ZeroMemory(&si, sizeof(si)); 
        si.cb = sizeof(si); 
        PROCESS_INFORMATION pi = detail::win32_start(entries[i].executable, entries[i].arguments, ctx.environment, sii, sio, sie, s); 

        if (!::CloseHandle(pi.hThread)) 
            boost::throw_exception(boost::system::system_error(boost::system::error_code(::GetLastError(), boost::system::get_system_category()), "boost::process::launch_pipeline: CloseHandle failed")); 

        cs.push_back(child(pi.dwProcessId, fhinvalid, fhstdout, fhstderr, detail::file_handle(pi.hProcess))); 
    } 
#endif 

    return cs; 
} 

/** 
 * Waits for a collection of children to terminate. 
 * 
 * Given a collection of Child objects (such as std::vector<child> or 
 * the convenience children type), waits for the termination of all of 
 * them. 
 * 
 * \remark Blocking remarks: This call blocks if any of the children 
 *         processes in the collection has not finalized execution and 
 *         waits until it terminates. 
 * 
 * \return The exit status of the first process that returns an error 
 *         code or, if all of them executed correctly, the exit status 
 *         of the last process in the collection. 
 */ 
template <class Children> 
inline status wait_children(Children &cs) 
{ 
    BOOST_ASSERT(cs.size() >= 2); 

    typename Children::iterator it = cs.begin(); 
    while (it != cs.end()) 
    { 
        const status s = it->wait(); 
        ++it; 
        if (it == cs.end()) 
            return s; 
        else if (!s.exited() || s.exit_status() != EXIT_SUCCESS) 
        { 
            while (it != cs.end()) 
            { 
                it->wait(); 
                ++it; 
            } 
            return s; 
        } 
    } 

    BOOST_ASSERT(false); 
    return cs.begin()->wait(); 
} 

} 
} 

#endif 
