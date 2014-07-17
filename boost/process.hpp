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
 * \file boost/process.hpp 
 * 
 * Convenience header that includes all other Boost.Process public header 
 * files. It is important to note that those headers that are specific to 
 * a given platform are only included if the library is being used in that 
 * same platform. 
 */ 

#ifndef BOOST_PROCESS_HPP 
#define BOOST_PROCESS_HPP 

#include <boost/process/config.hpp> 

#if defined(BOOST_POSIX_API) 
#  include <boost/process/posix_child.hpp> 
#  include <boost/process/posix_context.hpp> 
#  include <boost/process/posix_operations.hpp> 
#  include <boost/process/posix_status.hpp> 
#elif defined(BOOST_WINDOWS_API) 
#  include <boost/process/win32_child.hpp> 
#  include <boost/process/win32_context.hpp> 
#  include <boost/process/win32_operations.hpp> 
#else 
#  error "Unsupported platform." 
#endif 

#include <boost/process/child.hpp> 
#include <boost/process/context.hpp> 
#include <boost/process/environment.hpp> 
#include <boost/process/operations.hpp> 
#include <boost/process/pistream.hpp> 
#include <boost/process/postream.hpp> 
#include <boost/process/process.hpp> 
#include <boost/process/self.hpp> 
#include <boost/process/status.hpp> 
#include <boost/process/stream_behavior.hpp> 

#endif 
