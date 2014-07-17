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
 * \file boost/process/environment.hpp 
 * 
 * Includes the declaration of the environment class. 
 */ 

#ifndef BOOST_PROCESS_ENVIRONMENT_HPP 
#define BOOST_PROCESS_ENVIRONMENT_HPP 

#include <string> 
#include <map> 

namespace boost { 
namespace process { 

/** 
 * Representation of a process' environment variables. 
 * 
 * The environment is a map that stablishes an unidirectional 
 * association between variable names and their values and is 
 * represented by a string to string map. 
 * 
 * Variables may be defined to the empty string. Be aware that doing so 
 * is not portable: POSIX systems will treat such variables as being 
 * defined to the empty value, but Windows systems are not able to 
 * distinguish them from undefined variables. 
 * 
 * Neither POSIX nor Windows systems support a variable with no name. 
 * 
 * It is worthy to note that the environment is sorted alphabetically. 
 * This is provided for-free by the map container used to implement this 
 * type, and this behavior is required by Windows systems. 
 */ 
typedef std::map<std::string, std::string> environment; 

} 
} 

#endif 
