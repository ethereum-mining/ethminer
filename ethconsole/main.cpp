
#include <libjsconsole/JSRemoteConsole.h>

using namespace std;
using namespace dev;
using namespace dev::eth;

int main(int argc, char** argv)
{
	if (argc != 2)
	{
		cout << "You must provide remote url\n";
		cout << "eg:\n";
		cout << "./ethconsole http://localhost:8545\n";
		return 1;
	}

	JSRemoteConsole console(argv[1]);
	while (true)
		console.repl();

	return 0;
}