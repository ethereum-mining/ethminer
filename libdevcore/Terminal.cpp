/*
 Enable using colors in windows terminals which do not support ANSI
 escape codes.
*/
#ifdef _WIN32
#include <windows.h>
#include <iostream>
#include "Terminal.h"

// Define colors here since including windows.h is apparently not possible.
// They are defined in wincon.h.
#define FOREGROUND_BLUE      0x0001 // text color contains blue.
#define FOREGROUND_GREEN     0x0002 // text color contains green.
#define FOREGROUND_RED       0x0004 // text color contains red.
#define FOREGROUND_INTENSITY 0x0008 // text color is intensified.
#define BACKGROUND_BLUE      0x0010 // background color contains blue.
#define BACKGROUND_GREEN     0x0020 // background color contains green.
#define BACKGROUND_RED       0x0040 // background color contains red.
#define BACKGROUND_INTENSITY 0x0080 // background color is intensified.

#ifndef ENABLE_VIRTUAL_TERMINAL_PROCESSING
#define ENABLE_VIRTUAL_TERMINAL_PROCESSING 0x0004
#endif

enum EthForegroundColors
{
	FG_EthBlack = 0,
	FG_EthCoal = FOREGROUND_INTENSITY,
	FG_EthGray = FOREGROUND_BLUE | FOREGROUND_GREEN | FOREGROUND_RED,
	FG_EthWhite = FOREGROUND_BLUE | FOREGROUND_GREEN | FOREGROUND_RED | FOREGROUND_INTENSITY,
	FG_EthMaroon = FOREGROUND_RED,
	FG_EthRed = FOREGROUND_RED | FOREGROUND_INTENSITY,
	FG_EthGreen = FOREGROUND_GREEN,
	FG_EthLime = FOREGROUND_GREEN | FOREGROUND_INTENSITY,
	FG_EthOrange = FOREGROUND_GREEN | FOREGROUND_RED,
	FG_EthYellow = FOREGROUND_GREEN | FOREGROUND_RED | FOREGROUND_INTENSITY,
	FG_EthNavy = FOREGROUND_BLUE,
	FG_EthBlue = FOREGROUND_BLUE | FOREGROUND_INTENSITY,
	FG_EthViolet = FOREGROUND_BLUE | FOREGROUND_RED,
	FG_EthPurple = FOREGROUND_BLUE | FOREGROUND_RED | FOREGROUND_INTENSITY,
	FG_EthTeal = FOREGROUND_BLUE | FOREGROUND_GREEN,
	FG_EthCyan = FOREGROUND_BLUE | FOREGROUND_GREEN | FOREGROUND_INTENSITY
};

enum EthBackgroundColors
{
	BG_EthOnBlack = 0,
	BG_EthOnCoal = BACKGROUND_INTENSITY,
	BG_EthOnGray = BACKGROUND_BLUE | BACKGROUND_GREEN | BACKGROUND_RED,
	BG_EthOnWhite = BACKGROUND_BLUE | BACKGROUND_GREEN | BACKGROUND_RED | BACKGROUND_INTENSITY,
	BG_EthOnMaroon = BACKGROUND_RED,
	BG_EthOnRed = BACKGROUND_RED | BACKGROUND_INTENSITY,
	BG_EthOnGreen = BACKGROUND_GREEN,
	BG_EthOnLime = BACKGROUND_GREEN | BACKGROUND_INTENSITY,
	BG_EthOnOrange = BACKGROUND_GREEN | BACKGROUND_RED,
	BG_EthOnYellow = BACKGROUND_GREEN | BACKGROUND_RED | BACKGROUND_INTENSITY,
	BG_EthOnNavy = BACKGROUND_BLUE,
	BG_EthOnBlue = BACKGROUND_BLUE | BACKGROUND_INTENSITY,
	BG_EthOnViolet = BACKGROUND_BLUE | BACKGROUND_RED,
	BG_EthOnPurple = BACKGROUND_BLUE | BACKGROUND_RED | BACKGROUND_INTENSITY,
	BG_EthOnTeal = BACKGROUND_BLUE | BACKGROUND_GREEN,
	BG_EthOnCyan = BACKGROUND_BLUE | BACKGROUND_GREEN | BACKGROUND_INTENSITY
};

namespace dev
{
	namespace con {

		static WORD originalTextAttributes = 0;

		bool IsANSISupported()
		{
			HANDLE hConsole = GetStdHandle(STD_OUTPUT_HANDLE);
			DWORD consoleMode;
			if (GetConsoleMode(hConsole, &consoleMode))
			{
				if (consoleMode & ENABLE_VIRTUAL_TERMINAL_PROCESSING)
					return true;
			}
			return false;
		}

		void ResetTerminalColor()
		{
			HANDLE hConsole = GetStdHandle(STD_OUTPUT_HANDLE);
			if (originalTextAttributes)
				SetConsoleTextAttribute(hConsole, originalTextAttributes);
		}

		void SetTerminalForegroundColor(WORD wAttributes)
		{
			HANDLE hConsole = GetStdHandle(STD_OUTPUT_HANDLE);

			// Save original text attributes so we can revert
			CONSOLE_SCREEN_BUFFER_INFO consoleScreenBufferInfo;
			GetConsoleScreenBufferInfo(hConsole, &consoleScreenBufferInfo);
			WORD textAttributes;
			textAttributes = consoleScreenBufferInfo.wAttributes;
			if (!originalTextAttributes)
				originalTextAttributes = textAttributes;
			
			// Remove all foreground attributes
			textAttributes &= ~(FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_BLUE | FOREGROUND_INTENSITY);

			// And set the foreground we want
			textAttributes |= wAttributes;
			SetConsoleTextAttribute(hConsole, textAttributes);
		}

		void SetTerminalBackgroundColor(WORD wAttributes)
		{
			HANDLE hConsole = GetStdHandle(STD_OUTPUT_HANDLE);

			// Save original text attributes so we can revert
			CONSOLE_SCREEN_BUFFER_INFO consoleScreenBufferInfo;
			GetConsoleScreenBufferInfo(hConsole, &consoleScreenBufferInfo);
			WORD textAttributes;
			textAttributes = consoleScreenBufferInfo.wAttributes;
			if (!originalTextAttributes)
				originalTextAttributes = textAttributes;

			// Remove all background attributes
			textAttributes &= ~(BACKGROUND_RED | BACKGROUND_GREEN | BACKGROUND_BLUE | BACKGROUND_INTENSITY);

			// And set the background we want
			textAttributes |= wAttributes;
			SetConsoleTextAttribute(hConsole, textAttributes);
		}

		void HandleEscapeCode(std::string const& escapeCode)
		{
			// Reset
			if (escapeCode == EthReset)
				ResetTerminalColor();

			// Foreground
			else if (escapeCode == EthBlack || escapeCode == EthBlackBold)
				SetTerminalForegroundColor(FG_EthBlack);
			else if (escapeCode == EthCoal || escapeCode == EthCoalBold)
				SetTerminalForegroundColor(FG_EthCoal);
			else if (escapeCode == EthGray || escapeCode == EthGrayBold)
				SetTerminalForegroundColor(FG_EthGray);
			else if (escapeCode == EthWhite || escapeCode == EthWhiteBold)
				SetTerminalForegroundColor(FG_EthWhite);
			else if (escapeCode == EthMaroon || escapeCode == EthMaroonBold)
				SetTerminalForegroundColor(FG_EthMaroon);
			else if (escapeCode == EthRed || escapeCode == EthRedBold)
				SetTerminalForegroundColor(FG_EthRed);
			else if (escapeCode == EthGreen || escapeCode == EthGreenBold)
				SetTerminalForegroundColor(FG_EthGreen);
			else if (escapeCode == EthLime || escapeCode == EthLimeBold)
				SetTerminalForegroundColor(FG_EthLime);
			else if (escapeCode == EthOrange || escapeCode == EthOrangeBold)
				SetTerminalForegroundColor(FG_EthOrange);
			else if (escapeCode == EthYellow || escapeCode == EthYellowBold)
				SetTerminalForegroundColor(FG_EthYellow);
			else if (escapeCode == EthNavy || escapeCode == EthNavyBold)
				SetTerminalForegroundColor(FG_EthNavy);
			else if (escapeCode == EthBlue || escapeCode == EthBlueBold)
				SetTerminalForegroundColor(FG_EthBlue);
			else if (escapeCode == EthViolet || escapeCode == EthVioletBold)
				SetTerminalForegroundColor(FG_EthViolet);
			else if (escapeCode == EthPurple || escapeCode == EthPurpleBold)
				SetTerminalForegroundColor(FG_EthPurple);
			else if (escapeCode == EthTeal || escapeCode == EthTealBold)
				SetTerminalForegroundColor(FG_EthTeal);
			else if (escapeCode == EthCyan || escapeCode == EthCyanBold)
				SetTerminalForegroundColor(FG_EthCyan);

			// Background
			else if (escapeCode == EthOnBlack)
				SetTerminalBackgroundColor(BG_EthOnBlack);
			else if (escapeCode == EthOnCoal)
				SetTerminalBackgroundColor(BG_EthOnCoal);
			else if (escapeCode == EthOnGray)
				SetTerminalBackgroundColor(BG_EthOnGray);
			else if (escapeCode == EthOnWhite)
				SetTerminalBackgroundColor(BG_EthOnWhite);
			else if (escapeCode == EthOnMaroon)
				SetTerminalBackgroundColor(BG_EthOnMaroon);
			else if (escapeCode == EthOnRed)
				SetTerminalBackgroundColor(BG_EthOnRed);
			else if (escapeCode == EthOnGreen)
				SetTerminalBackgroundColor(BG_EthOnGreen);
			else if (escapeCode == EthOnLime)
				SetTerminalBackgroundColor(BG_EthOnLime);
			else if (escapeCode == EthOnOrange)
				SetTerminalBackgroundColor(BG_EthOnOrange);
			else if (escapeCode == EthOnYellow)
				SetTerminalBackgroundColor(BG_EthOnYellow);
			else if (escapeCode == EthOnNavy)
				SetTerminalBackgroundColor(BG_EthOnNavy);
			else if (escapeCode == EthOnBlue)
				SetTerminalBackgroundColor(BG_EthOnBlue);
			else if (escapeCode == EthOnViolet)
				SetTerminalBackgroundColor(BG_EthOnViolet);
			else if (escapeCode == EthOnPurple)
				SetTerminalBackgroundColor(BG_EthOnPurple);
			else if (escapeCode == EthOnTeal)
				SetTerminalBackgroundColor(BG_EthOnTeal);
			else if (escapeCode == EthOnCyan)
				SetTerminalBackgroundColor(BG_EthOnCyan);
		}

		void HandleLogOutput(std::string const& _s)
		{
			if (IsANSISupported())
			{
				std::cerr << _s.c_str() << std::endl;
				return;
			}

			std::string s = _s;

			while (true)
			{
				size_t offsetStart = s.find_first_of('\x1b');
				size_t offsetEnd = s.find_first_of('m');
				if (offsetStart != std::string::npos && offsetEnd != std::string::npos)
				{
					if (offsetStart < offsetEnd)
					{
						// Print beginning of the buffer until escape code
						if (offsetStart)
							std::cerr << s.substr(0, offsetStart).c_str();

						// Handle escape code
						std::string escapeCode = s.substr(offsetStart, offsetEnd - offsetStart + 1);
						HandleEscapeCode(escapeCode);

						// Continue with the remaining buffer
						s = s.substr(offsetEnd + 1);
					}
					// If we found ESC after m, it just means the buffer starts with m.
					else
					{
						// Print beginning of the buffer until escape code
						if (offsetStart)
							std::cerr << s.substr(0, offsetStart).c_str();

						// Continue with the remaining buffer
						s = s.substr(offsetStart);
					}
				}
				else
				{
					std::cerr << s.c_str() << std::endl;
					break;
				}
			}
		}
	}
}

#endif
