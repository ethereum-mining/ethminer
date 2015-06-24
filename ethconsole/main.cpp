
#include <string>
#include <libjsconsole/JSRemoteConsole.h>

using namespace std;
using namespace dev;
using namespace dev::eth;

int main(int argc, char** argv)
{
	string remote;
	if (argc != 2)
	{
		cout << "remote url not provided\n";
		cout << "using default:\n";
		cout << "./ethconsole http://localhost:8545\n";
		remote = "http://localhost:8545\n";
	}
	else
		remote = argv[1];

	JSRemoteConsole console(remote);
	while (true)
		console.readExpression();

	return 0;
}