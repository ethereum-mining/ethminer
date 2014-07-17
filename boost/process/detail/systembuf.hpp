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
 * \file boost/process/detail/systembuf.hpp 
 * 
 * Includes the declaration of the systembuf class. This file is for 
 * internal usage only and must not be included by the library user. 
 */ 

#ifndef BOOST_PROCESS_DETAIL_SYSTEMBUF_HPP 
#define BOOST_PROCESS_DETAIL_SYSTEMBUF_HPP 

#include <boost/process/config.hpp> 

#if defined(BOOST_POSIX_API) 
#  include <sys/types.h> 
#  include <unistd.h> 
#elif defined(BOOST_WINDOWS_API) 
#  include <windows.h> 
#else 
#  error "Unsupported platform." 
#endif 

#include <boost/noncopyable.hpp> 
#include <boost/scoped_array.hpp> 
#include <boost/assert.hpp> 
#include <streambuf> 
#include <cstddef> 

namespace boost { 
namespace process { 

class postream; 

namespace detail { 

/** 
 * std::streambuf implementation for system file handles. 
 * 
 * systembuf provides a std::streambuf implementation for system file 
 * handles. Contrarywise to file_handle, this class does \b not take 
 * ownership of the native file handle; this should be taken care of 
 * somewhere else. 
 * 
 * This class follows the expected semantics of a std::streambuf object. 
 * However, it is not copyable to avoid introducing inconsistences with 
 * the on-disk file and the in-memory buffers. 
 */ 
class systembuf : public std::streambuf, public boost::noncopyable 
{ 
    friend class ::boost::process::postream; 

public: 
#if defined(BOOST_PROCESS_DOXYGEN) 
    /** 
     * Opaque name for the native handle type. 
     */ 
    typedef NativeHandleType handle_type; 
#elif defined(BOOST_POSIX_API) 
    typedef int handle_type; 
#elif defined(BOOST_WINDOWS_API) 
    typedef HANDLE handle_type; 
#endif 

    /** 
     * Constructs a new systembuf for the given file handle. 
     * 
     * This constructor creates a new systembuf object that reads or 
     * writes data from/to the \a h native file handle. This handle 
     * is \b not owned by the created systembuf object; the code 
     * should take care of it externally. 
     * 
     * This class buffers input and output; the buffer size may be 
     * tuned through the \a bufsize parameter, which defaults to 8192 
     * bytes. 
     * 
     * \see pistream and postream 
     */ 
    explicit systembuf(handle_type h, std::size_t bufsize = 8192) 
        : handle_(h), 
        bufsize_(bufsize), 
        read_buf_(new char[bufsize]), 
        write_buf_(new char[bufsize]) 
    { 
#if defined(BOOST_POSIX_API) 
        BOOST_ASSERT(handle_ >= 0); 
#elif defined(BOOST_WINDOWS_API) 
        BOOST_ASSERT(handle_ != INVALID_HANDLE_VALUE); 
#endif 
        BOOST_ASSERT(bufsize_ > 0); 

        setp(write_buf_.get(), write_buf_.get() + bufsize_); 
    } 

protected: 
    /** 
     * Reads new data from the native file handle. 
     * 
     * This operation is called by input methods when there is no more 
     * data in the input buffer. The function fills the buffer with new 
     * data, if available. 
     * 
     * \pre All input positions are exhausted (gptr() >= egptr()). 
     * \post The input buffer has new data, if available. 
     * \returns traits_type::eof() if a read error occurrs or there are 
     *          no more data to be read. Otherwise returns 
     *          traits_type::to_int_type(*gptr()). 
     */ 
    virtual int_type underflow() 
    { 
        BOOST_ASSERT(gptr() >= egptr()); 

        bool ok; 
#if defined(BOOST_POSIX_API) 
        ssize_t cnt = ::read(handle_, read_buf_.get(), bufsize_); 
        ok = (cnt != -1 && cnt != 0); 
#elif defined(BOOST_WINDOWS_API) 
        DWORD cnt; 
        BOOL res = ::ReadFile(handle_, read_buf_.get(), bufsize_, &cnt, NULL); 
        ok = (res && cnt > 0); 
#endif 

        if (!ok) 
            return traits_type::eof(); 
        else 
        { 
            setg(read_buf_.get(), read_buf_.get(), read_buf_.get() + cnt); 
            return traits_type::to_int_type(*gptr()); 
        } 
    } 

    /** 
     * Makes room in the write buffer for additional data. 
     * 
     * This operation is called by output methods when there is no more 
     * space in the output buffer to hold a new element. The function 
     * first flushes the buffer's contents to disk and then clears it to 
     * leave room for more characters. The given \a c character is 
     * stored at the beginning of the new space. 
     * 
     * \pre All output positions are exhausted (pptr() >= epptr()). 
     * \post The output buffer has more space if no errors occurred 
     *       during the write to disk. 
     * \post *(pptr() - 1) is \a c. 
     * \returns traits_type::eof() if a write error occurrs. Otherwise 
     *          returns traits_type::not_eof(c). 
     */ 
    virtual int_type overflow(int c) 
    { 
        BOOST_ASSERT(pptr() >= epptr()); 

        if (sync() == -1) 
            return traits_type::eof(); 

        if (!traits_type::eq_int_type(c, traits_type::eof())) 
        { 
            traits_type::assign(*pptr(), c); 
            pbump(1); 
        } 

        return traits_type::not_eof(c); 
    } 

    /** 
     * Flushes the output buffer to disk. 
     * 
     * Synchronizes the systembuf buffers with the contents of the file 
     * associated to this object through the native file handle. The 
     * output buffer is flushed to disk and cleared to leave new room 
     * for more data. 
     * 
     * \returns 0 on success, -1 if an error occurred. 
     */ 
    virtual int sync() 
    { 
#if defined(BOOST_POSIX_API) 
        ssize_t cnt = pptr() - pbase(); 
#elif defined(BOOST_WINDOWS_API) 
        long cnt = pptr() - pbase(); 
#endif 

        bool ok; 
#if defined(BOOST_POSIX_API) 
        ok = ::write(handle_, pbase(), cnt) == cnt; 
#elif defined(BOOST_WINDOWS_API) 
        DWORD rcnt; 
        BOOL res = ::WriteFile(handle_, pbase(), cnt, &rcnt, NULL); 
        ok = (res && static_cast<long>(rcnt) == cnt); 
#endif 

        if (ok) 
            pbump(-cnt); 
        return ok ? 0 : -1; 
    } 

private: 
    /** 
     * Native file handle used by the systembuf object. 
     */ 
    handle_type handle_; 

    /** 
     * Internal buffer size used during read and write operations. 
     */ 
    std::size_t bufsize_; 

    /** 
     * Internal buffer used during read operations. 
     */ 
    boost::scoped_array<char> read_buf_; 

    /** 
     * Internal buffer used during write operations. 
     */ 
    boost::scoped_array<char> write_buf_; 
}; 

} 
} 
} 

#endif 
