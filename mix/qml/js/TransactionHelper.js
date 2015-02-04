Qt.include("QEtherHelper.js")

function defaultTransaction()
{
	return {
		value: createEther("0", QEther.Wei),
		functionId: "",
		gas: createEther("125000", QEther.Wei),
		gasPrice: createEther("100000", QEther.Wei),
		executeConstructor: false,
		parameters: {}
	};
}
