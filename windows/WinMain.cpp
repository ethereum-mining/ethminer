// http://www.flipcode.com/archives/WinMain_Command_Line_Parser.shtml
// COTD Entry submitted by Max McGuire [amcguire@andrew.cmu.edu]

#include <windows.h>

extern int main(int argc, char* argv[]);

int WINAPI WinMain(HINSTANCE instance, HINSTANCE prev_instance, char* command_line, int show_command)
{
	int    argc;
	char** argv;
	char*  arg;
	int    index;
	int    result;

	// count the arguments
	argc = 1;
	arg = command_line;

	while (arg[0] != 0)
	{
		while (arg[0] != 0 && arg[0] == ' ')
		{
			arg++;
		}
		if (arg[0] != 0)
		{
			argc++;
			while (arg[0] != 0 && arg[0] != ' ')
			{
				arg++;
			}
		}
	}

	// tokenize the arguments
	argv = (char**)malloc(argc * sizeof(char*));
	arg = command_line;
	index = 1;

	while (arg[0] != 0)
	{
		while (arg[0] != 0 && arg[0] == ' ')
		{
			arg++;
		}
		if (arg[0] != 0)
		{
			argv[index] = arg;
			index++;
			while (arg[0] != 0 && arg[0] != ' ')
			{
				arg++;
			}
			if (arg[0] != 0)
			{
				arg[0] = 0;
				arg++;
			}
		}
	}

	// put the program name into argv[0]
	char filename[_MAX_PATH];
	GetModuleFileName(NULL, filename, _MAX_PATH);
	argv[0] = filename;

	// call the user specified main function    
	result = main(argc, argv);
	free(argv);
	return result;
}
