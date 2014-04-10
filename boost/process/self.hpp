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
 * \file boost/process/self.hpp 
 * 
 * Includes the declaration of the self class. 
 */ 

#ifndef BOOST_PROCESS_SELF_HPP 
#define BOOST_PROCESS_SELF_HPP 

#include <boost/process/config.hpp> 

#if defined(BOOST_POSIX_API) 
#  include <unistd.h> 
#elif defined(BOOST_WINDOWS_API) 
#  include <windows.h> 
#else 
#  error "Unsupported platform." 
#endif 

#include <boost/process/process.hpp> 
#include <boost/process/environment.hpp> 
#include <boost/system/system_error.hpp> 
#include <boost/throw_exception.hpp> 
#include <boost/noncopyable.hpp> 
#include <string> 

#if defined(BOOST_POSIX_API) 
extern "C" 
{ 
    extern char **environ; 
} 
#endif 

namespace boost { 
namespace process { 

/** 
 * Generic implementation of the Process concept. 
 * 
 * The self singleton provides access to the current process. 
 */ 
class self : public process, boost::noncopyable 
{ 
public: 
    /** 
     * Returns the self instance representing the caller's process. 
     */ 
    static self &get_instance() 
    { 
        static self *instance = 0; 
        if (!instance) 
            instance = new self; 
        return *instance; 
    } 

    /** 
     * Returns the current environment. 
     * 
     * Returns the current process' environment variables. Modifying the 
     * returned object has no effect on the current environment. 
     */ 
    static environment get_environment() 
    { 
        environment e; 

#if defined(BOOST_POSIX_API) 
        char **env = ::environ; 
        while (*env) 
        { 
            std::string s = *env; 
            std::string::size_type pos = s.find('='); 
            e.insert(boost::process::environment::value_type(s.substr(0, pos), s.substr(pos + 1))); 
            ++env; 
        } 
#elif defined(BOOST_WINDOWS_API) 
#ifdef GetEnvironmentStrings 
#undef GetEnvironmentStrings 
#endif 
        char *environ = ::GetEnvironmentStrings(); 
        if (!environ) 
            boost::throw_exception(boost::system::system_error(boost::system::error_code(::GetLastError(), boost::system::get_system_category()), "boost::process::self::get_environment: GetEnvironmentStrings failed")); 
        try 
        { 
            char *env = environ; 
            while (*env) 
            { 
                std::string s = env; 
                std::string::size_type pos = s.find('='); 
                e.insert(boost::process::environment::value_type(s.substr(0, pos), s.substr(pos + 1))); 
                env += s.size() + 1; 
            } 
        } 
        catch (...) 
        { 
            ::FreeEnvironmentStringsA(environ); 
            throw; 
        } 
        ::FreeEnvironmentStringsA(environ); 
#endif 

        return e; 
    } 

private: 
    /** 
     * Constructs a new self object. 
     * 
     * Creates a new self object that represents the current process. 
     */ 
    self() : 
#if defined(BOOST_POSIX_API) 
       process(::getpid()) 
#elif defined(BOOST_WINDOWS_API) 
       process(::GetCurrentProcessId()) 
#endif 
       { 
       } 
}; 

} 
} 

#endif 
