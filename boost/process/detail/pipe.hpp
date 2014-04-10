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
 * \file boost/process/detail/pipe.hpp 
 * 
 * Includes the declaration of the pipe class. This file is for 
 * internal usage only and must not be included by the library user. 
 */ 

#ifndef BOOST_PROCESS_DETAIL_PIPE_HPP 
#define BOOST_PROCESS_DETAIL_PIPE_HPP 

#include <boost/process/config.hpp> 

#if defined(BOOST_POSIX_API) 
#  include <unistd.h> 
#  include <cerrno> 
#elif defined(BOOST_WINDOWS_API) 
#  if defined(BOOST_PROCESS_WINDOWS_USE_NAMED_PIPE) 
#    include <boost/lexical_cast.hpp> 
#    include <string> 
#  endif 
#  include <windows.h> 
#else 
#  error "Unsupported platform." 
#endif 

#include <boost/process/detail/file_handle.hpp> 
#include <boost/system/system_error.hpp> 
#include <boost/throw_exception.hpp> 

namespace boost { 
namespace process { 
namespace detail { 

/** 
 * Simple RAII model for anonymous pipes. 
 * 
 * The pipe class is a simple RAII model for anonymous pipes. It 
 * provides a portable constructor that allocates a new %pipe and creates 
 * a pipe object that owns the two file handles associated to it: the 
 * read end and the write end. 
 * 
 * These handles can be retrieved for modification according to 
 * file_handle semantics. Optionally, their ownership can be transferred 
 * to external \a file_handle objects which comes handy when the two 
 * ends need to be used in different places (i.e. after a POSIX fork() 
 * system call). 
 * 
 * Pipes can be copied following the same semantics as file handles. 
 * In other words, copying a %pipe object invalidates the source one. 
 * 
 * \see file_handle 
 */ 
class pipe 
{ 
public: 
    /** 
     * Creates a new %pipe. 
     * 
     * The default pipe constructor allocates a new anonymous %pipe 
     * and assigns its ownership to the created pipe object. On Windows 
     * when the macro BOOST_PROCESS_WINDOWS_USE_NAMED_PIPE is defined 
     * a named pipe is created. This is required if asynchronous I/O 
     * should be used as asynchronous I/O is only supported by named 
     * pipes on Windows. 
     * 
     * \throw boost::system::system_error If the anonymous %pipe 
     *        creation fails. 
     */ 
    pipe() 
    { 
        file_handle::handle_type hs[2]; 

#if defined(BOOST_POSIX_API) 
        if (::pipe(hs) == -1) 
            boost::throw_exception(boost::system::system_error(boost::system::error_code(errno, boost::system::get_system_category()), "boost::process::detail::pipe::pipe: pipe(2) failed")); 
#elif defined(BOOST_WINDOWS_API) 
        SECURITY_ATTRIBUTES sa; 
        ZeroMemory(&sa, sizeof(sa)); 
        sa.nLength = sizeof(sa); 
        sa.lpSecurityDescriptor = NULL; 
        sa.bInheritHandle = FALSE; 

#  if defined(BOOST_PROCESS_WINDOWS_USE_NAMED_PIPE) 
        static unsigned int nextid = 0; 
        std::string pipe = "\\\\.\\pipe\\boost_process_" + boost::lexical_cast<std::string>(::GetCurrentProcessId()) + "_" + boost::lexical_cast<std::string>(nextid++); 
        hs[0] = ::CreateNamedPipeA(pipe.c_str(), PIPE_ACCESS_INBOUND | FILE_FLAG_OVERLAPPED, 0, 1, 8192, 8192, 0, &sa); 
        if (hs[0] == INVALID_HANDLE_VALUE) 
            boost::throw_exception(boost::system::system_error(::GetLastError(), boost::system::system_category, "boost::process::detail::pipe::pipe: CreateNamedPipe failed")); 
        hs[1] = ::CreateFileA(pipe.c_str(), GENERIC_WRITE, 0, NULL, OPEN_EXISTING, FILE_FLAG_OVERLAPPED, NULL); 
        if (hs[1] == INVALID_HANDLE_VALUE) 
            boost::throw_exception(boost::system::system_error(::GetLastError(), boost::system::system_category, "boost::process::detail::pipe::pipe: CreateFile failed")); 

        OVERLAPPED overlapped; 
        ZeroMemory(&overlapped, sizeof(overlapped)); 
        overlapped.hEvent = ::CreateEvent(NULL, TRUE, FALSE, NULL); 
        if (!overlapped.hEvent) 
            boost::throw_exception(boost::system::system_error(::GetLastError(), boost::system::system_category, "boost::process::detail::pipe::pipe: CreateEvent failed")); 
        BOOL b = ::ConnectNamedPipe(hs[0], &overlapped); 
        if (!b) 
        { 
            if (::GetLastError() == ERROR_IO_PENDING) 
            { 
                if (::WaitForSingleObject(overlapped.hEvent, INFINITE) == WAIT_FAILED) 
                { 
                    ::CloseHandle(overlapped.hEvent); 
                    boost::throw_exception(boost::system::system_error(::GetLastError(), boost::system::system_category, "boost::process::detail::pipe::pipe: WaitForSingleObject failed")); 
                } 
            } 
            else if (::GetLastError() != ERROR_PIPE_CONNECTED) 
            { 
                ::CloseHandle(overlapped.hEvent); 
                boost::throw_exception(boost::system::system_error(::GetLastError(), boost::system::system_category, "boost::process::detail::pipe::pipe: ConnectNamedPipe failed")); 
            } 
        } 
        ::CloseHandle(overlapped.hEvent); 
#  else 
        if (!::CreatePipe(&hs[0], &hs[1], &sa, 0)) 
            boost::throw_exception(boost::system::system_error(::GetLastError(), boost::system::system_category, "boost::process::detail::pipe::pipe: CreatePipe failed")); 
#  endif 
#endif 

        read_end_ = file_handle(hs[0]); 
        write_end_ = file_handle(hs[1]); 
    } 

    /** 
     * Returns the %pipe's read end file handle. 
     * 
     * Obtains a reference to the %pipe's read end file handle. Care 
     * should be taken to not duplicate the returned object if ownership 
     * shall remain to the %pipe. 
     * 
     * Duplicating the returned object invalidates its corresponding file 
     * handle in the %pipe. 
     * 
     * \return A reference to the %pipe's read end file handle. 
     */ 
    file_handle &rend() 
    { 
        return read_end_; 
    } 

    /** 
     * Returns the %pipe's write end file handle. 
     * 
     * Obtains a reference to the %pipe's write end file handle. Care 
     * should be taken to not duplicate the returned object if ownership 
     * shall remain to the %pipe. 
     * 
     * Duplicating the returned object invalidates its corresponding file 
     * handle in the %pipe. 
     * 
     * \return A reference to the %pipe's write end file handle. 
     */ 
    file_handle &wend() 
    { 
        return write_end_; 
    } 

private: 
    /** 
     * The %pipe's read end file handle. 
     */ 
    file_handle read_end_; 

    /** 
     * The %pipe's write end file handle. 
     */ 
    file_handle write_end_; 
}; 

} 
} 
} 

#endif 
