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
 * \file boost/process/context.hpp 
 * 
 * Includes the declaration of the context class and several accessory 
 * base classes. 
 */ 

#ifndef BOOST_PROCESS_CONTEXT_HPP 
#define BOOST_PROCESS_CONTEXT_HPP 

#include <boost/process/config.hpp> 

#if defined(BOOST_POSIX_API) 
#  include <boost/scoped_array.hpp> 
#  include <cerrno> 
#  include <unistd.h> 
#elif defined(BOOST_WINDOWS_API) 
#  include <windows.h> 
#else 
#  error "Unsupported platform." 
#endif 

#include <boost/process/environment.hpp> 
#include <boost/process/stream_behavior.hpp> 
#include <boost/system/system_error.hpp> 
#include <boost/throw_exception.hpp> 
#include <boost/assert.hpp> 
#include <string> 
#include <vector> 

namespace boost { 
namespace process { 

/** 
 * Base context class that defines the child's work directory. 
 * 
 * Base context class that defines the necessary fields to configure a 
 * child's work directory. This class is useless on its own because no 
 * function in the library will accept it as a valid Context 
 * implementation. 
 */ 
template <class Path> 
class basic_work_directory_context 
{ 
public: 
    /** 
     * Constructs a new work directory context. 
     * 
     * Constructs a new work directory context making the work directory 
     * described by the new object point to the caller's current working 
     * directory. 
     */ 
    basic_work_directory_context() 
    { 
#if defined(BOOST_POSIX_API) 
        errno = 0; 
        long size = ::pathconf(".", _PC_PATH_MAX); 
        if (size == -1 && errno) 
            boost::throw_exception(boost::system::system_error(boost::system::error_code(errno, boost::system::get_system_category()), "boost::process::basic_work_directory_context::basic_work_directory_context: pathconf(2) failed")); 
        else if (size == -1) 
            size = BOOST_PROCESS_POSIX_PATH_MAX; 
        boost::scoped_array<char> cwd(new char[size]); 
        if (!::getcwd(cwd.get(), size)) 
            boost::throw_exception(boost::system::system_error(boost::system::error_code(errno, boost::system::get_system_category()), "boost::process::basic_work_directory_context::basic_work_directory_context: getcwd(2) failed")); 
        work_directory = cwd.get(); 
#elif defined(BOOST_WINDOWS_API) 
        char cwd[MAX_PATH]; 
        if (!::GetCurrentDirectoryA(sizeof(cwd), cwd)) 
            boost::throw_exception(boost::system::system_error(boost::system::error_code(::GetLastError(), boost::system::get_system_category()), "boost::process::basic_work_directory_context::basic_work_directory_context: GetCurrentDirectory failed")); 
        work_directory = cwd; 
#endif 
        BOOST_ASSERT(!work_directory.empty()); 
    } 

    /** 
     * The process' initial work directory. 
     * 
     * The work directory is the directory in which the process starts 
     * execution. 
     */ 
    Path work_directory; 
}; 

/** 
 * Base context class that defines the child's environment. 
 * 
 * Base context class that defines the necessary fields to configure a 
 * child's environment variables. This class is useless on its own 
 * because no function in the library will accept it as a valid Context 
 * implementation. 
 */ 
class environment_context 
{ 
public: 
    /** 
     * The process' environment. 
     * 
     * Contains the list of environment variables, alongside with their 
     * values, that will be passed to the spawned child process. 
     */ 
    boost::process::environment environment; 
}; 

/** 
 * Process startup execution context. 
 * 
 * The context class groups all the parameters needed to configure a 
 * process' environment during its creation. 
 */ 
template <class Path> 
class basic_context : public basic_work_directory_context<Path>, public environment_context 
{ 
public: 
    /** 
     * Child's stdin behavior. 
     */ 
    stream_behavior stdin_behavior; 

    /** 
     * Child's stdout behavior. 
     */ 
    stream_behavior stdout_behavior; 

    /** 
     * Child's stderr behavior. 
     */ 
    stream_behavior stderr_behavior; 
}; 

typedef basic_context<std::string> context; 

/** 
 * Represents a child process in a pipeline. 
 * 
 * This convenience class is a triplet that holds all the data required 
 * to spawn a new child process in a pipeline. 
 */ 
template <class Executable, class Arguments, class Context> 
class basic_pipeline_entry 
{ 
public: 
    /** 
     * The executable to launch. 
     */ 
    Executable executable; 

    /** 
     * The set of arguments to pass to the executable. 
     */ 
    Arguments arguments; 

    /** 
     * The child's execution context. 
     */ 
    Context context; 

    /** 
     * The type of the Executable concept used in this template 
     * instantiation. 
     */ 
    typedef Executable executable_type; 

    /** 
     * The type of the Arguments concept used in this template 
     * instantiation. 
     */ 
    typedef Arguments arguments_type; 

    /** 
     * The type of the Context concept used in this template 
     * instantiation. 
     */ 
    typedef Context context_type; 

    /** 
     * Constructs a new pipeline_entry object. 
     * 
     * Given the executable, set of arguments and execution triplet, 
     * constructs a new pipeline_entry object that holds the three 
     * values. 
     */ 
    basic_pipeline_entry(const Executable &exe, const Arguments &args, const Context &ctx) 
        : executable(exe), 
        arguments(args), 
        context(ctx) 
    { 
    } 
}; 

/** 
 * Default instantiation of basic_pipeline_entry. 
 */ 
typedef basic_pipeline_entry<std::string, std::vector<std::string>, context> pipeline_entry; 

} 
} 

#endif 
