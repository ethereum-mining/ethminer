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
 * \file boost/process/postream.hpp 
 * 
 * Includes the declaration of the postream class. 
 */ 

#ifndef BOOST_PROCESS_POSTREAM_HPP 
#define BOOST_PROCESS_POSTREAM_HPP 

#include <boost/process/detail/file_handle.hpp> 
#include <boost/process/detail/systembuf.hpp> 
#include <boost/noncopyable.hpp> 
#include <ostream> 

namespace boost { 
namespace process { 

/** 
 * Child process' input stream. 
 * 
 * The postream class represents an input communication channel with the 
 * child process. The child process reads data from this stream and the 
 * parent process can write to it through the postream object. In other 
 * words, from the child's point of view, the communication channel is an 
 * input one, but from the parent's point of view it is an output one; 
 * hence the confusing postream name. 
 * 
 * postream objects cannot be copied because they own the file handle 
 * they use to communicate with the child and because they buffer data 
 * that flows through the communication channel. 
 * 
 * A postream object behaves as a std::ostream stream in all senses. 
 * The class is only provided because it must provide a method to let 
 * the caller explicitly close the communication channel. 
 * 
 * \remark Blocking remarks: Functions that write data to this 
 *         stream can block if the associated file handle blocks during 
 *         the write. As this class is used to communicate with child 
 *         processes through anonymous pipes, the most typical blocking 
 *         condition happens when the child is not processing the data 
 *         in the pipe's system buffer. When this happens, the buffer 
 *         eventually fills up and the system blocks until the reader 
 *         consumes some data, leaving some new room. 
 */ 
class postream : public std::ostream, public boost::noncopyable 
{ 
public: 
    /** 
     * Creates a new process' input stream. 
     * 
     * Given a file handle, this constructor creates a new postream 
     * object that owns the given file handle \a fh. Ownership of 
     * \a fh is transferred to the created postream object. 
     * 
     * \pre \a fh is valid. 
     * \post \a fh is invalid. 
     * \post The new postream object owns \a fh. 
     */ 
    explicit postream(detail::file_handle &fh) 
        : std::ostream(0), 
        handle_(fh), 
        systembuf_(handle_.get()) 
    { 
        rdbuf(&systembuf_); 
    } 

    /** 
     * Returns the file handle managed by this stream. 
     * 
     * The file handle must not be copied. Copying invalidates 
     * the source file handle making the postream unusable. 
     */ 
    detail::file_handle &handle() 
    { 
        return handle_; 
    } 

    /** 
     * Closes the file handle managed by this stream. 
     * 
     * Explicitly closes the file handle managed by this stream. This 
     * function can be used by the user to tell the child process there 
     * is no more data to send. 
     */ 
    void close() 
    { 
        systembuf_.sync(); 
        handle_.close(); 
    } 

private: 
    /** 
     * The file handle managed by this stream. 
     */ 
    detail::file_handle handle_; 

    /** 
     * The systembuf object used to manage this stream's data. 
     */ 
    detail::systembuf systembuf_; 
}; 

} 
} 

#endif 
