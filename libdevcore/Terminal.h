#pragma once

namespace dev
{
namespace con
{

#ifdef _WIN32

#define EthReset ""       // Text Reset

// Regular Colors
#define EthBlack ""        // Black
#define EthRed ""          // Red
#define EthGreen ""        // Green
#define EthYellow ""       // Yellow
#define EthBlue ""         // Blue
#define EthPurple ""       // Purple
#define EthCyan ""         // Cyan
#define EthWhite ""        // White

// Bold
#define EthBlackB ""       // Black
#define EthRedB ""         // Red
#define EthGreenB ""       // Green
#define EthYellowB ""      // Yellow
#define EthBlueB ""        // Blue
#define EthPurpleB ""      // Purple
#define EthCyanB ""        // Cyan
#define EthWhiteB ""       // White

// Underline
#define EthBlackU ""       // Black
#define EthRedU ""         // Red
#define EthGreenU ""       // Green
#define EthYellowU ""      // Yellow
#define EthBlueU ""        // Blue
#define EthPurpleU ""      // Purple
#define EthCyanU ""        // Cyan
#define EthWhiteU ""       // White

// Background
#define EthBlackOn ""       // Black
#define EthRedOn ""         // Red
#define EthGreenOn ""       // Green
#define EthYellowOn ""      // Yellow
#define EthBlueOn ""        // Blue
#define EthPurpleOn ""      // Purple
#define EthCyanOn ""        // Cyan
#define EthWhiteOn ""       // White

// High Intensity
#define EthCoal ""       // Black
#define EthRedI ""         // Red
#define EthGreenI ""       // Green
#define EthYellowI ""      // Yellow
#define EthBlueI ""        // Blue
#define EthPurpleI ""      // Purple
#define EthCyanI ""        // Cyan
#define EthWhiteI ""       // White

// Bold High Intensity
#define EthBlackBI ""      // Black
#define EthRedBI ""        // Red
#define EthGreenBI ""      // Green
#define EthYellowBI ""     // Yellow
#define EthBlueBI ""       // Blue
#define EthPurpleBI ""     // Purple
#define EthCyanBI ""       // Cyan
#define EthWhiteBI ""      // White

// High Intensity backgrounds
#define EthBlackOnI ""   // Black
#define EthRedOnI ""     // Red
#define EthGreenOnI ""   // Green
#define EthYellowOnI ""  // Yellow
#define EthBlueOnI ""    // Blue
#define EthPurpleOnI ""  // Purple
#define EthCyanOnI ""    // Cyan
#define EthWhiteOnI ""   // White

#else

#define EthReset "\x1b[0m"       // Text Reset

// Regular Colors
#define EthBlack "\x1b[30m"        // Black
#define EthCoal "\x1b[90m"       // Black
#define EthGray "\x1b[37m"        // White
#define EthWhite "\x1b[97m"       // White
#define EthRed "\x1b[31m"          // Red
#define EthGreen "\x1b[32m"        // Green
#define EthYellow "\x1b[33m"       // Yellow
#define EthBlue "\x1b[34m"         // Blue
#define EthPurple "\x1b[35m"       // Purple
#define EthCyan "\x1b[36m"         // Cyan
// High Intensity
#define EthRedI "\x1b[91m"         // Red
#define EthGreenI "\x1b[92m"       // Green
#define EthYellowI "\x1b[93m"      // Yellow
#define EthBlueI "\x1b[94m"        // Blue
#define EthPurpleI "\x1b[95m"      // Purple
#define EthCyanI "\x1b[96m"        // Cyan

// Bold
#define EthBlackB "\x1b[1;30m"       // Black
#define EthCoalB "\x1b[1;90m"      // Black
#define EthGrayB "\x1b[1;37m"       // White
#define EthWhiteB "\x1b[1;97m"      // White
#define EthRedB "\x1b[1;31m"         // Red
#define EthGreenB "\x1b[1;32m"       // Green
#define EthYellowB "\x1b[1;33m"      // Yellow
#define EthBlueB "\x1b[1;34m"        // Blue
#define EthPurpleB "\x1b[1;35m"      // Purple
#define EthCyanB "\x1b[1;36m"        // Cyan
// Bold High Intensity
#define EthRedBI "\x1b[1;91m"        // Red
#define EthGreenBI "\x1b[1;92m"      // Green
#define EthYellowBI "\x1b[1;93m"     // Yellow
#define EthBlueBI "\x1b[1;94m"       // Blue
#define EthPurpleBI "\x1b[1;95m"     // Purple
#define EthCyanBI "\x1b[1;96m"       // Cyan

// Background
#define EthBlackOn "\x1b[40m"       // Black
#define EthCoalOn "\x1b[100m"   // Black
#define EthGrayOn "\x1b[47m"       // White
#define EthWhiteOn "\x1b[107m"   // White
#define EthRedOn "\x1b[41m"         // Red
#define EthGreenOn "\x1b[42m"       // Green
#define EthYellowOn "\x1b[43m"      // Yellow
#define EthBlueOn "\x1b[44m"        // Blue
#define EthPurpleOn "\x1b[45m"      // Purple
#define EthCyanOn "\x1b[46m"        // Cyan
// High Intensity backgrounds
#define EthRedOnI "\x1b[101m"     // Red
#define EthGreenOnI "\x1b[102m"   // Green
#define EthYellowOnI "\x1b[103m"  // Yellow
#define EthBlueOnI "\x1b[104m"    // Blue
#define EthPurpleOnI "\x1b[105m"  // Purple
#define EthCyanOnI "\x1b[106m"    // Cyan

// Underline
#define EthBlackU "\x1b[4;30m"       // Black
#define EthRedU "\x1b[4;31m"         // Red
#define EthGreenU "\x1b[4;32m"       // Green
#define EthYellowU "\x1b[4;33m"      // Yellow
#define EthBlueU "\x1b[4;34m"        // Blue
#define EthPurpleU "\x1b[4;35m"      // Purple
#define EthCyanU "\x1b[4;36m"        // Cyan
#define EthWhiteU "\x1b[4;37m"       // White

#endif

}

}
