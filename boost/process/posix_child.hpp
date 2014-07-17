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
 * \file boost/process/posix_child.hpp 
 * 
 * Includes the declaration of the posix_child class. 
 */ 

#ifndef BOOST_PROCESS_POSIX_CHILD_HPP 
#define BOOST_PROCESS_POSIX_CHILD_HPP 

#include <boost/process/child.hpp> 
#include <boost/process/pistream.hpp> 
#include <boost/process/postream.hpp> 
#include <boost/process/detail/pipe.hpp> 
#include <boost/process/detail/posix_ops.hpp> 
#include <boost/process/detail/stream_info.hpp> 
#include <boost/shared_ptr.hpp> 
#include <boost/assert.hpp> 
#include <map> 
#include <unistd.h> 

namespace boost { 
namespace process { 

/** 
 * POSIX implementation of the Child concept. 
 * 
 * The posix_child class implements the Child concept in a POSIX 
 * operating system. 
 * 
 * A POSIX child differs from a regular child (represented by a 
 * child object) in that it supports more than three communication 
 * channels with its parent. These channels are identified by regular 
 * file descriptors (integers). 
 * 
 * This class is built on top of the generic child so as to allow its 
 * trivial adoption. When a program is changed to use the POSIX-specific 
 * context (posix_context), it will most certainly need to migrate its 
 * use of the child class to posix_child. Doing so is only a matter of 
 * redefining the appropriate object and later using the required extra 
 * features: there should be no need to modify the existing code (e.g. 
 * method calls) in any other way. 
 */ 
class posix_child : public child 
{ 
public: 
    /** 
     * Gets a reference to the child's input stream \a desc. 
     * 
     * Returns a reference to a postream object that represents one of 
     * the multiple input communication channels with the child process. 
     * The channel is identified by \a desc as seen from the child's 
     * point of view. The parent can use the returned stream to send 
     * data to the child. 
     * 
     * Giving this function the STDIN_FILENO constant (defined in 
     * unistd.h) is a synonym for the get_stdin() call inherited from 
     * child. 
     */ 
    postream &get_input(int desc) const 
    { 
        if (desc == STDIN_FILENO) 
            return posix_child::get_stdin(); 
        else 
        { 
            input_map_t::const_iterator it = input_map_.find(desc); 
            BOOST_ASSERT(it != input_map_.end()); 
            return *it->second; 
        } 
    } 

    /** 
     * Gets a reference to the child's output stream \a desc. 
     * 
     * Returns a reference to a pistream object that represents one of 
     * the multiple output communication channels with the child process. 
     * The channel is identified by \a desc as seen from the child's 
     * point of view. The parent can use the returned stream to retrieve 
     * data from the child. 
     * 
     * Giving this function the STDOUT_FILENO or STDERR_FILENO constants 
     * (both defined in unistd.h) are synonyms for the get_stdout() and 
     * get_stderr() calls inherited from child, respectively. 
     */ 
    pistream &get_output(int desc) const 
    { 
        if (desc == STDOUT_FILENO) 
            return posix_child::get_stdout(); 
        else if (desc == STDERR_FILENO) 
            return posix_child::get_stderr(); 
        else 
        { 
            output_map_t::const_iterator it = output_map_.find(desc); 
            BOOST_ASSERT(it != output_map_.end()); 
            return *it->second; 
        } 
    } 

    /** 
     * Constructs a new POSIX child object representing a just 
     * spawned child process. 
     * 
     * Creates a new child object that represents the just spawned process 
     * \a id. 
     * 
     * The \a infoin and \a infoout maps contain the pipes used to handle 
     * the redirections of the child process; at the moment, no other 
     * stream_info types are supported. If the launcher was asked to 
     * redirect any of the three standard flows, their pipes must be 
     * present in these maps. 
     */ 
    posix_child(id_type id, detail::info_map &infoin, detail::info_map &infoout) 
        : child(id, 
        detail::posix_info_locate_pipe(infoin, STDIN_FILENO, false), 
        detail::posix_info_locate_pipe(infoout, STDOUT_FILENO, true), 
        detail::posix_info_locate_pipe(infoout, STDERR_FILENO, true)) 
    { 
        for (detail::info_map::iterator it = infoin.begin(); it != infoin.end(); ++it) 
        { 
            detail::stream_info &si = it->second; 
            if (si.type_ == detail::stream_info::use_pipe) 
            { 
                BOOST_ASSERT(si.pipe_->wend().valid()); 
                boost::shared_ptr<postream> st(new postream(si.pipe_->wend())); 
                input_map_.insert(input_map_t::value_type(it->first, st)); 
            } 
        } 

        for (detail::info_map::iterator it = infoout.begin(); it != infoout.end(); ++it) 
        { 
            detail::stream_info &si = it->second; 
            if (si.type_ == detail::stream_info::use_pipe) 
            { 
                BOOST_ASSERT(si.pipe_->rend().valid()); 
                boost::shared_ptr<pistream> st(new pistream(si.pipe_->rend())); 
                output_map_.insert(output_map_t::value_type(it->first, st)); 
            } 
        } 
    } 

private: 
    /** 
     * Maps child's file descriptors to postream objects. 
     */ 
    typedef std::map<int, boost::shared_ptr<postream> > input_map_t; 

    /** 
     * Contains all relationships between child's input file 
     * descriptors and their corresponding postream objects. 
     */ 
    input_map_t input_map_; 

    /** 
     * Maps child's file descriptors to pistream objects. 
     */ 
    typedef std::map<int, boost::shared_ptr<pistream> > output_map_t; 

    /** 
     * Contains all relationships between child's output file 
     * descriptors and their corresponding pistream objects. 
     */ 
    output_map_t output_map_; 
}; 

} 
} 

#endif 
