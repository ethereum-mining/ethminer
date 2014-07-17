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
 * \file boost/process/stream_behavior.hpp 
 * 
 * Includes the declaration of the stream_behavior class and associated 
 * free functions. 
 */ 

#ifndef BOOST_PROCESS_STREAM_BEHAVIOR_HPP 
#define BOOST_PROCESS_STREAM_BEHAVIOR_HPP 

#include <boost/process/config.hpp> 

namespace boost { 
namespace process { 

namespace detail { 
    struct stream_info; 
} 

/** 
 * Describes the possible states for a communication stream. 
 */ 
class stream_behavior 
{ 
public: 
    friend struct detail::stream_info; 
    friend stream_behavior capture_stream(); 
    friend stream_behavior close_stream(); 
    friend stream_behavior inherit_stream(); 
    friend stream_behavior redirect_stream_to_stdout(); 
    friend stream_behavior silence_stream(); 
#if defined(BOOST_POSIX_API) || defined(BOOST_PROCESS_DOXYGEN) 
    friend stream_behavior posix_redirect_stream(int to); 
#endif 

    /** 
     * Describes the possible states for a communication stream. 
     */ 
    enum type 
    { 
        /** 
         * The child's stream is connected to the parent by using an 
         * anonymous pipe so that they can send and receive data to/from 
         * each other. 
         */ 
        capture, 

        /** 
         * The child's stream is closed upon startup so that it will not 
         * have any access to it. 
         */ 
        close, 

        /** 
         * The child's stream is connected to the same stream used by the 
         * parent. In other words, the corresponding parent's stream is 
         * inherited. 
         */ 
        inherit, 

        /** 
         * The child's stream is connected to child's standard output. 
         * This is typically used when configuring the standard error 
         * stream. 
         */ 
        redirect_to_stdout, 

        /** 
         * The child's stream is redirected to a null device so that its 
         * input is always zero or its output is lost, depending on 
         * whether the stream is an input or an output one. It is 
         * important to notice that this is different from close because 
         * the child is still able to write data. If we closed, e.g. 
         * stdout, the child might not work at all! 
         */ 
        silence, 

#if defined(BOOST_POSIX_API) || defined(BOOST_PROCESS_DOXYGEN) 
        /** 
         * The child redirects the stream's output to the provided file 
         * descriptor. This is a generalization of the portable 
         * redirect_to_stdout behavior. 
         */ 
        posix_redirect 
#endif 
    }; 

    /** 
     * Constructs a new stream behavior of type close. 
     * 
     * The public constructor creates a new stream behavior that defaults 
     * to the close behavior. In general, you will want to use the 
     * available free functions to construct a stream behavior (including 
     * the close one). 
     */ 
    stream_behavior() 
        : type_(stream_behavior::close) 
    { 
    } 

    /** 
     * Returns this stream's behavior type. 
     */ 
    type get_type() const 
    { 
        return type_; 
    } 

private: 
    /** 
     * Constructs a new stream behavior of type \a t. 
     * 
     * Constructs a new stream behavior of type \a t. It is the 
     * responsibility of the caller to fill in any other attributes 
     * required by the specified type, if any. 
     */ 
    stream_behavior(type t) 
        : type_(t) 
    { 
    } 

    /** 
     * This stream's behavior type. 
     */ 
    type type_; 

#if defined(BOOST_POSIX_API) || defined(BOOST_PROCESS_DOXYGEN) 
    /** 
     * File descriptor the stream is redirected to. 
     */ 
    int desc_to_; 
#endif 
}; 

/** 
 * Creates a new stream_behavior of type stream_behavior::capture. 
 * 
 * Creates a new stream_behavior of type stream_behavior::capture, 
 * meaning that the child's stream is connected to the parent by using an 
 * anonymous pipe so that they can send and receive data to/from each 
 * other. 
 */ 
inline stream_behavior capture_stream() 
{ 
    return stream_behavior(stream_behavior::capture); 
} 

/** 
 * Creates a new stream_behavior of type stream_behavior::close. 
 * 
 * Creates a new stream_behavior of type stream_behavior::close, 
 * meaning that the child's stream is closed upon startup so that it 
 * will not have any access to it. 
 */ 
inline stream_behavior close_stream() 
{ 
    return stream_behavior(stream_behavior::close); 
} 

/** 
 * Creates a new stream_behavior of type stream_behavior::inherit. 
 * 
 * Creates a new stream_behavior of type stream_behavior::inherit, 
 * meaning that the child's stream is connected to the same stream used 
 * by the parent. In other words, the corresponding parent's stream is 
 * inherited. 
 */ 
inline stream_behavior inherit_stream() 
{ 
    return stream_behavior(stream_behavior::inherit); 
} 

/** 
 * Creates a new stream_behavior of type 
 * stream_behavior::redirect_to_stdout. 
 * 
 * Creates a new stream_behavior of type 
 * stream_behavior::redirect_to_stdout, meaning that the child's stream is 
 * connected to child's standard output. This is typically used when 
 * configuring the standard error stream. 
 */ 
inline stream_behavior redirect_stream_to_stdout() 
{ 
    return stream_behavior(stream_behavior::redirect_to_stdout); 
} 

/** 
 * Creates a new stream_behavior of type stream_behavior::silence. 
 * 
 * Creates a new stream_behavior of type stream_behavior::silence, 
 * meaning that the child's stream is redirected to a null device so that 
 * its input is always zero or its output is lost, depending on whether 
 * the stream is an input or an output one. It is important to notice 
 * that this is different from close because the child is still able to 
 * write data. If we closed, e.g. stdout, the child might not work at 
 * all! 
 */ 
inline stream_behavior silence_stream() 
{ 
    return stream_behavior(stream_behavior::silence); 
} 

#if defined(BOOST_POSIX_API) || defined(BOOST_PROCESS_DOXYGEN) 
/** 
 * Creates a new stream_behavior of type stream_behavior::posix_redirect. 
 * 
 * Creates a new stream_behavior of type stream_behavior::posix_redirect, 
 * meaning that the child's stream is redirected to the \a to child's 
 * file descriptor. This is a generalization of the portable 
 * redirect_stream_to_stdout() behavior. 
 */ 
inline stream_behavior posix_redirect_stream(int to) 
{ 
    stream_behavior sb(stream_behavior::posix_redirect); 
    sb.desc_to_ = to; 
    return sb; 
} 
#endif 

} 
} 

#endif 
