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
#define EthLime ""       // Green
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
#define EthMaroon "\x1b[31m"          // Red
#define EthRed "\x1b[91m"         // Red
#define EthGreen "\x1b[32m"        // Green
#define EthLime "\x1b[92m"       // Green
#define EthOrange "\x1b[33m"       // Yellow
#define EthYellow "\x1b[93m"      // Yellow
#define EthNavy "\x1b[34m"         // Blue
#define EthBlue "\x1b[94m"        // Blue
#define EthViolet "\x1b[35m"       // Purple
#define EthPurple "\x1b[95m"      // Purple
#define EthTeal "\x1b[36m"         // Cyan
#define EthCyan "\x1b[96m"        // Cyan

#define EthBlackBold "\x1b[1;30m"       // Black
#define EthCoalBold "\x1b[1;90m"      // Black
#define EthGrayBold "\x1b[1;37m"       // White
#define EthWhiteBold "\x1b[1;97m"      // White
#define EthMaroonBold "\x1b[1;31m"         // Red
#define EthRedBold "\x1b[1;91m"        // Red
#define EthGreenBold "\x1b[1;32m"       // Green
#define EthLimeBold "\x1b[1;92m"      // Green
#define EthOrangeBold "\x1b[1;33m"      // Yellow
#define EthYellowBold "\x1b[1;93m"     // Yellow
#define EthNavyBold "\x1b[1;34m"        // Blue
#define EthBlueBold "\x1b[1;94m"       // Blue
#define EthVioletBold "\x1b[1;35m"      // Purple
#define EthPurpleBold "\x1b[1;95m"     // Purple
#define EthTealBold "\x1b[1;36m"        // Cyan
#define EthCyanBold "\x1b[1;96m"       // Cyan

// Background
#define EthOnBlack "\x1b[40m"       // Black
#define EthOnCoal "\x1b[100m"   // Black
#define EthOnGray "\x1b[47m"       // White
#define EthOnWhite "\x1b[107m"   // White
#define EthOnMaroon "\x1b[41m"         // Red
#define EthOnRed "\x1b[101m"     // Red
#define EthOnGreen "\x1b[42m"       // Green
#define EthOnLime "\x1b[102m"   // Green
#define EthOnOrange "\x1b[43m"      // Yellow
#define EthOnYellow "\x1b[103m"  // Yellow
#define EthOnNavy "\x1b[44m"        // Blue
#define EthOnBlue "\x1b[104m"    // Blue
#define EthOnViolet "\x1b[45m"      // Purple
#define EthOnPurple "\x1b[105m"  // Purple
#define EthOnTeal "\x1b[46m"        // Cyan
#define EthOnCyan "\x1b[106m"    // Cyan

// Underline
#define EthBlackUnder "\x1b[4;30m"       // Black
#define EthGrayUnder "\x1b[4;37m"       // White
#define EthMaroonUnder "\x1b[4;31m"      // Red
#define EthGreenUnder "\x1b[4;32m"       // Green
#define EthOrangeUnder "\x1b[4;33m"      // Yellow
#define EthNavyUnder "\x1b[4;34m"        // Blue
#define EthVioletUnder "\x1b[4;35m"      // Purple
#define EthTealUnder "\x1b[4;36m"        // Cyan

#endif

}

}
