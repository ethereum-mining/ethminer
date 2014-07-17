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
 * \file boost/process/detail/stream_info.hpp 
 * 
 * Provides the definition of the stream_info structure. 
 */ 

#ifndef BOOST_PROCESS_DETAIL_STREAM_INFO_HPP 
#define BOOST_PROCESS_DETAIL_STREAM_INFO_HPP 

#include <boost/process/config.hpp> 

#if defined(BOOST_POSIX_API) 
#  include <unistd.h> 
#elif defined(BOOST_WINDOWS_API) 
#else 
#  error "Unsupported platform." 
#endif 

#include <boost/process/stream_behavior.hpp> 
#include <boost/process/detail/file_handle.hpp> 
#include <boost/process/detail/pipe.hpp> 
#include <boost/optional.hpp> 
#include <boost/assert.hpp> 
#include <string> 

namespace boost { 
namespace process { 
namespace detail { 

/** 
 * Configuration data for a file descriptor. 
 * 
 * This convenience structure provides a compact way to pass information 
 * around on how to configure a file descriptor. It is a lower-level 
 * representation of stream_behavior, as it can hold the same information 
 * but in a way that can be used by the underlying operating system. 
 */ 
struct stream_info 
{ 
    /** 
     * Supported stream types. 
     */ 
    enum type 
    { 
        /** 
         * Matches stream_behavior::close. 
         */ 
        close, 

        /** 
         * Matches stream_behavior::inherit. 
         */ 
        inherit, 

        /** 
         * Matches stream_behavior::redirect_to_stdout and 
         * stream_behavior::posix_redirect. 
         */ 
        redirect, 

        /** 
         * Matches stream_behavior::silence. 
         */ 
        use_file, 

        /** 
         * TODO: Matches nothing yet ... 
         */ 
        use_handle, 

        /** 
         * Matches stream_behavior::capture. 
         */ 
        use_pipe 
    }; 

    /** 
     * Stream type. 
     */ 
    type type_; 

    /** 
     * Descriptor to use when stream type is set to \a redirect. 
     */ 
    int desc_to_; 

    /** 
     * File to use when stream type is set to \a use_file. 
     */ 
    std::string file_; 

    /** 
     * Handle to use when stream type is set to \a use_handle. 
     */ 
    file_handle handle_; 

    /** 
     * Pipe to use when stream type is set to \a use_pipe. 
     */ 
    boost::optional<pipe> pipe_; 

    /** 
     * Constructs a new stream_info object. 
     */ 
    stream_info(const stream_behavior &sb, bool out) 
    { 
        switch (sb.type_) 
        { 
        case stream_behavior::close: 
            { 
                type_ = close; 
                break; 
            } 
        case stream_behavior::inherit: 
            { 
                type_ = inherit; 
                break; 
            } 
        case stream_behavior::redirect_to_stdout: 
            { 
                type_ = redirect; 
#if defined(BOOST_POSIX_API) 
                desc_to_ = STDOUT_FILENO; 
#elif defined(BOOST_WINDOWS_API) 
                desc_to_ = 1; 
#endif 
                break; 
            } 
#if defined(BOOST_POSIX_API) 
        case stream_behavior::posix_redirect: 
            { 
                type_ = redirect; 
                desc_to_ = sb.desc_to_; 
                break; 
            } 
#endif 
        case stream_behavior::silence: 
            { 
                type_ = use_file; 
#if defined(BOOST_POSIX_API) 
                file_ = out ? "/dev/null" : "/dev/zero"; 
#elif defined(BOOST_WINDOWS_API) 
                file_ = "NUL"; 
#endif 
                break; 
            } 
        case stream_behavior::capture: 
            { 
                type_ = use_pipe; 
                pipe_ = pipe(); 
                break; 
            } 
        default: 
            { 
                BOOST_ASSERT(false); 
            } 
        } 
    } 
}; 

} 
} 
} 

#endif 
