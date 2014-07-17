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
 * \file boost/process/detail/posix_ops.hpp 
 * 
 * Provides some convenience functions to start processes under POSIX 
 * operating systems. 
 */ 

#ifndef BOOST_PROCESS_DETAIL_POSIX_OPS_HPP 
#define BOOST_PROCESS_DETAIL_POSIX_OPS_HPP 

#include <boost/process/environment.hpp> 
#include <boost/process/detail/file_handle.hpp> 
#include <boost/process/detail/pipe.hpp> 
#include <boost/process/detail/stream_info.hpp> 
#include <boost/scoped_array.hpp> 
#include <boost/assert.hpp> 
#include <boost/system/system_error.hpp> 
#include <boost/throw_exception.hpp> 
#include <map> 
#include <utility> 
#include <string> 
#include <cerrno> 
#include <cstdlib> 
#include <cstring> 
#include <fcntl.h> 
#include <unistd.h> 

namespace boost { 
namespace process { 
namespace detail { 

/** 
 * Converts the command line to an array of C strings. 
 * 
 * Converts the command line's list of arguments to the format expected 
 * by the \a argv parameter in the POSIX execve() system call. 
 * 
 * This operation is only available on POSIX systems. 
 * 
 * \return The first argument of the pair is an integer that indicates 
 *         how many strings are stored in the second argument. The 
 *         second argument is a NULL-terminated, dynamically allocated 
 *         array of dynamically allocated strings holding the arguments 
 *         to the executable. The caller is responsible of freeing them. 
 */ 
template <class Arguments> 
inline std::pair<std::size_t, char**> collection_to_posix_argv(const Arguments &args) 
{ 
    std::size_t nargs = args.size(); 
    BOOST_ASSERT(nargs > 0); 

    char **argv = new char*[nargs + 1]; 
    typename Arguments::size_type i = 0; 
    for (typename Arguments::const_iterator it = args.begin(); it != args.end(); ++it) 
    { 
        argv[i] = new char[it->size() + 1]; 
        std::strncpy(argv[i], it->c_str(), it->size() + 1); 
        ++i; 
    } 
    argv[nargs] = 0; 

    return std::pair<std::size_t, char**>(nargs, argv); 
} 

/** 
 * Converts an environment to a char** table as used by execve(). 
 * 
 * Converts the environment's contents to the format used by the 
 * execve() system call. The returned char** array is allocated 
 * in dynamic memory; the caller must free it when not used any 
 * more. Each entry is also allocated in dynamic memory and is a 
 * NULL-terminated string of the form var=value; these must also be 
 * released by the caller. 
 * 
 * \return A dynamically allocated char** array that represents 
 *         the environment's content. Each array entry is a 
 *         NULL-terminated string of the form var=value. 
 */ 
inline char **environment_to_envp(const environment &env) 
{ 
    char **envp = new char*[env.size() + 1]; 

    environment::size_type i = 0; 
    for (environment::const_iterator it = env.begin(); it != env.end(); ++it) 
    { 
        std::string s = it->first + "=" + it->second; 
        envp[i] = new char[s.size() + 1]; 
        std::strncpy(envp[i], s.c_str(), s.size() + 1); 
        ++i; 
    } 
    envp[i] = 0; 

    return envp; 
} 

/** 
 * Holds a mapping between native file descriptors and their corresponding 
 * pipes to set up communication between the parent and the %child process. 
 */ 
typedef std::map<int, stream_info> info_map; 

/** 
 * Helper class to configure a POSIX %child. 
 * 
 * This helper class is used to hold all the attributes that configure a 
 * new POSIX %child process and to centralize all the actions needed to 
 * make them effective. 
 * 
 * All its fields are public for simplicity. It is only intended for 
 * internal use and it is heavily coupled with the Context 
 * implementations. 
 */ 
struct posix_setup 
{ 
    /** 
     * The work directory. 
     * 
     * This string specifies the directory in which the %child process 
     * starts execution. It cannot be empty. 
     */ 
    std::string work_directory; 

    /** 
     * The chroot directory, if any. 
     * 
     * Specifies the directory in which the %child process is chrooted 
     * before execution. Empty if this feature is not desired. 
     */ 
    std::string chroot; 

    /** 
     * The user credentials. 
     * 
     * UID that specifies the user credentials to use to run the %child 
     * process. Defaults to the current UID. 
     */ 
    uid_t uid; 

    /** 
     * The effective user credentials. 
     * 
     * EUID that specifies the effective user credentials to use to run 
     * the %child process. Defaults to the current EUID. 
     */ 
    uid_t euid; 

    /** 
     * The group credentials. 
     * 
     * GID that specifies the group credentials to use to run the %child 
     * process. Defaults to the current GID. 
     */ 
    gid_t gid; 

    /** 
     * The effective group credentials. 
     * 
     * EGID that specifies the effective group credentials to use to run 
     * the %child process. Defaults to the current EGID. 
     */ 
    gid_t egid; 

    /** 
     * Creates a new properties set. 
     * 
     * Creates a new object that has sensible default values for all the 
     * properties. 
     */ 
    posix_setup() 
        : uid(::getuid()), 
        euid(::geteuid()), 
        gid(::getgid()), 
        egid(::getegid()) 
    { 
    } 

    /** 
     * Sets up the execution environment. 
     * 
     * Modifies the current execution environment (that of the %child) so 
     * that the properties become effective. 
     * 
     * \throw boost::system::system_error If any error ocurred during 
     *        environment configuration. The child process should abort 
     *        execution if this happens because its start conditions 
     *        cannot be met. 
     */ 
    void operator()() const 
    { 
        if (!chroot.empty() && ::chroot(chroot.c_str()) == -1) 
            boost::throw_exception(boost::system::system_error(boost::system::error_code(errno, boost::system::get_system_category()), "boost::process::detail::posix_setup: chroot(2) failed")); 

        if (gid != ::getgid() && ::setgid(gid) == -1) 
            boost::throw_exception(boost::system::system_error(boost::system::error_code(errno, boost::system::get_system_category()), "boost::process::detail::posix_setup: setgid(2) failed")); 

        if (egid != ::getegid() && ::setegid(egid) == -1) 
            boost::throw_exception(boost::system::system_error(boost::system::error_code(errno, boost::system::get_system_category()), "boost::process::detail::posix_setup: setegid(2) failed")); 

        if (uid != ::getuid() && ::setuid(uid) == -1) 
            boost::throw_exception(boost::system::system_error(boost::system::error_code(errno, boost::system::get_system_category()), "boost::process::detail::posix_setup: setuid(2) failed")); 

        if (euid != ::geteuid() && ::seteuid(euid) == -1) 
            boost::throw_exception(boost::system::system_error(boost::system::error_code(errno, boost::system::get_system_category()), "boost::process::detail::posix_setup: seteuid(2) failed")); 

        BOOST_ASSERT(!work_directory.empty()); 
        if (::chdir(work_directory.c_str()) == -1) 
            boost::throw_exception(boost::system::system_error(boost::system::error_code(errno, boost::system::get_system_category()), "boost::process::detail::posix_setup: chdir(2) failed")); 
    } 
}; 

/** 
 * Configures child process' input streams. 
 * 
 * Sets up the current process' input streams to behave according to the 
 * information in the \a info map. \a closeflags is modified to reflect 
 * those descriptors that should not be closed because they where modified 
 * by the function. 
 * 
 * Modifies the current execution environment, so this should only be run 
 * on the child process after the fork(2) has happened. 
 * 
 * \throw boost::system::system_error If any error occurs during the 
 *        configuration. 
 */ 
inline void setup_input(info_map &info, bool *closeflags, int maxdescs) 
{ 
    for (info_map::iterator it = info.begin(); it != info.end(); ++it) 
    { 
        int d = it->first; 
        stream_info &si = it->second; 

        BOOST_ASSERT(d < maxdescs); 
        closeflags[d] = false; 

        switch (si.type_) 
        { 
        case stream_info::use_file: 
            { 
                int fd = ::open(si.file_.c_str(), O_RDONLY); 
                if (fd == -1) 
                    boost::throw_exception(boost::system::system_error(boost::system::error_code(errno, boost::system::get_system_category()), "boost::process::detail::setup_input: open(2) of " + si.file_ + " failed")); 
                if (fd != d) 
                { 
                    file_handle h(fd); 
                    h.posix_remap(d); 
                    h.release(); 
                } 
                break; 
            } 
        case stream_info::use_handle: 
            { 
                if (si.handle_.get() != d) 
                    si.handle_.posix_remap(d); 
                break; 
            } 
        case stream_info::use_pipe: 
            { 
                si.pipe_->wend().close(); 
                if (d != si.pipe_->rend().get()) 
                    si.pipe_->rend().posix_remap(d); 
                break; 
            } 
        default: 
            { 
                BOOST_ASSERT(si.type_ == stream_info::inherit); 
                break; 
            } 
        } 
    } 
} 

/** 
 * Configures child process' output streams. 
 * 
 * Sets up the current process' output streams to behave according to the 
 * information in the \a info map. \a closeflags is modified to reflect 
 * those descriptors that should not be closed because they where 
 * modified by the function. 
 * 
 * Modifies the current execution environment, so this should only be run 
 * on the child process after the fork(2) has happened. 
 * 
 * \throw boost::system::system_error If any error occurs during the 
 *        configuration. 
 */ 
inline void setup_output(info_map &info, bool *closeflags, int maxdescs) 
{ 
    for (info_map::iterator it = info.begin(); it != info.end(); ++it) 
    { 
        int d = it->first; 
        stream_info &si = it->second; 

        BOOST_ASSERT(d < maxdescs); 
        closeflags[d] = false; 

        switch (si.type_) 
        { 
        case stream_info::redirect: 
            { 
                break; 
            } 
        case stream_info::use_file: 
            { 
                int fd = ::open(si.file_.c_str(), O_WRONLY); 
                if (fd == -1) 
                    boost::throw_exception(boost::system::system_error(boost::system::error_code(errno, boost::system::get_system_category()), "boost::process::detail::setup_output: open(2) of " + si.file_ + " failed")); 
                if (fd != d) 
                { 
                    file_handle h(fd); 
                    h.posix_remap(d); 
                    h.release(); 
                } 
                break; 
            } 
        case stream_info::use_handle: 
            { 
                if (si.handle_.get() != d) 
                    si.handle_.posix_remap(d); 
                break; 
            } 
        case stream_info::use_pipe: 
            { 
                si.pipe_->rend().close(); 
                if (d != si.pipe_->wend().get()) 
                    si.pipe_->wend().posix_remap(d); 
                break; 
            } 
        default: 
            { 
                BOOST_ASSERT(si.type_ == stream_info::inherit); 
                break; 
            } 
        } 
    } 

    for (info_map::const_iterator it = info.begin(); it != info.end(); ++it) 
    { 
        int d = it->first; 
        const stream_info &si = it->second; 

        if (si.type_ == stream_info::redirect) 
            file_handle::posix_dup(si.desc_to_, d).release(); 
    } 
} 

/** 
 * Starts a new child process in a POSIX operating system. 
 * 
 * This helper functions is provided to simplify the Context's task when 
 * it comes to starting up a new process in a POSIX operating system. 
 * The function hides all the details of the fork/exec pair of calls as 
 * well as all the setup of communication pipes and execution environment. 
 * 
 * \param exe The executable to spawn the child process. 
 * \param args The arguments for the executable. 
 * \param env The environment variables that the new child process 
 *        receives. 
 * \param infoin A map describing all input file descriptors to be 
 *        redirected. 
 * \param infoout A map describing all output file descriptors to be 
 *        redirected. 
 * \param setup A helper object used to configure the child's execution 
 *        environment. 
 * \return The new process' PID. The caller is responsible of creating 
 *         an appropriate Child representation for it. 
 */ 
template <class Executable, class Arguments> 
inline pid_t posix_start(const Executable &exe, const Arguments &args, const environment &env, info_map &infoin, info_map &infoout, const posix_setup &setup) 
{ 
    pid_t pid = ::fork(); 
    if (pid == -1) 
        boost::throw_exception(boost::system::system_error(boost::system::error_code(errno, boost::system::get_system_category()), "boost::process::detail::posix_start: fork(2) failed")); 
    else if (pid == 0) 
    { 
#if defined(F_MAXFD) 
        int maxdescs = ::fcntl(-1, F_MAXFD, 0); 
        if (maxdescs == -1) 
            maxdescs = ::sysconf(_SC_OPEN_MAX); 
#else 
        int maxdescs = ::sysconf(_SC_OPEN_MAX); 
#endif 
        if (maxdescs == -1) 
            maxdescs = 1024; 
        try 
        { 
            boost::scoped_array<bool> closeflags(new bool[maxdescs]); 
            for (int i = 0; i < maxdescs; ++i) 
                closeflags[i] = true; 

            setup_input(infoin, closeflags.get(), maxdescs); 
            setup_output(infoout, closeflags.get(), maxdescs); 

            for (int i = 0; i < maxdescs; ++i) 
            { 
                if (closeflags[i]) 
                    ::close(i); 
            } 

            setup(); 
        } 
        catch (const boost::system::system_error &e) 
        { 
            ::write(STDERR_FILENO, e.what(), std::strlen(e.what())); 
            ::write(STDERR_FILENO, "\n", 1); 
            std::exit(EXIT_FAILURE); 
        } 

        std::pair<std::size_t, char**> argcv = collection_to_posix_argv(args); 
        char **envp = environment_to_envp(env); 

        ::execve(exe.c_str(), argcv.second, envp); 
        boost::system::system_error e(boost::system::error_code(errno, boost::system::get_system_category()), "boost::process::detail::posix_start: execve(2) failed"); 

        for (std::size_t i = 0; i < argcv.first; ++i) 
            delete[] argcv.second[i]; 
        delete[] argcv.second; 

        for (std::size_t i = 0; i < env.size(); ++i) 
            delete[] envp[i]; 
        delete[] envp; 

        ::write(STDERR_FILENO, e.what(), std::strlen(e.what())); 
        ::write(STDERR_FILENO, "\n", 1); 
        std::exit(EXIT_FAILURE); 
    } 

    BOOST_ASSERT(pid > 0); 

    for (info_map::iterator it = infoin.begin(); it != infoin.end(); ++it) 
    { 
        stream_info &si = it->second; 
        if (si.type_ == stream_info::use_pipe) 
            si.pipe_->rend().close(); 
    } 

    for (info_map::iterator it = infoout.begin(); it != infoout.end(); ++it) 
    { 
        stream_info &si = it->second; 
        if (si.type_ == stream_info::use_pipe) 
            si.pipe_->wend().close(); 
    } 

    return pid; 
} 

/** 
 * Locates a communication pipe and returns one of its endpoints. 
 * 
 * Given a \a info map, and a file descriptor \a desc, searches for its 
 * communicataion pipe in the map and returns one of its endpoints as 
 * indicated by the \a out flag. This is intended to be used by a 
 * parent process after a fork(2) call. 
 * 
 * \pre If the info map contains the given descriptor, it is configured 
 *      to use a pipe. 
 * \post The info map does not contain the given descriptor. 
 * \return If the file descriptor is found in the map, returns the pipe's 
 *         read end if out is true; otherwise its write end. If the 
 *         descriptor is not found returns an invalid file handle. 
 */ 
inline file_handle posix_info_locate_pipe(info_map &info, int desc, bool out) 
{ 
    file_handle fh; 

    info_map::iterator it = info.find(desc); 
    if (it != info.end()) 
    { 
        stream_info &si = it->second; 
        if (si.type_ == stream_info::use_pipe) 
        { 
            fh = out ? si.pipe_->rend().release() : si.pipe_->wend().release(); 
            BOOST_ASSERT(fh.valid()); 
        } 
        info.erase(it); 
    } 

    return fh; 
} 

} 
} 
} 

#endif 
