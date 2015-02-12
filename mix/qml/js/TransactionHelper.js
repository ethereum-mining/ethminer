Qt.include("QEtherHelper.js")

function defaultTransaction()
{
	return {
		value: createEther("0", QEther.Wei),
		functionId: "",
		gas: createBigInt("125000"),
		gasPrice: createEther("100000", QEther.Wei),
		executeConstructor: false,
		parameters: {}
	};
}
