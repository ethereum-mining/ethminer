#include "RLP.h"
#include "BlockInfo.h"
using namespace std;
using namespace eth;

void populateAndVerify(bytesConstRef _block, u256 _number)
{
	number = _number;

	RLP root(_block);
	try
	{
		RLP header = root[0];
		hash = eth::sha256(_block);
		parentHash = header[0].toFatInt();
		sha256Uncles = header[1].toFatInt();
		coinbaseAddress = header[2].toFatInt();
		sha256Transactions = header[3].toFatInt();
		difficulty = header[4].toFatInt();
		timestamp = header[5].toFatInt();
		nonce = header[6].toFatInt();
	}
	catch (RLP::BadCast)
	{
		throw InvalidBlockFormat();
	}

	if (sha256Transactions != sha256(root[1].data()))
		throw InvalidTransactionsHash();

	if (sha256Uncles != sha256(root[2].data()))
		throw InvalidUnclesHash();

	// TODO: check timestamp.
	// TODO: check difficulty against timestamp.
	// TODO: check proof of work.

	// TODO: check each transaction.
}
