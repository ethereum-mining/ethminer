Qt.include("QEtherHelper.js")

function defaultTransaction()
{
	return {
		value: createEther("0", QEther.Wei),
		functionId: "",
		gas: createBigInt("250000"),
		gasAuto: true,
		gasPrice: createEther("100000", QEther.Wei),
		parameters: {},
		stdContract: false,
		isContractCreation: true,
		label: "",
		isFunctionCall: true,
		saveStatus: true
	};
}

function rpcCall(requests, callBack, error)
{
	var jsonRpcUrl = "http://localhost:8545";
	var rpcRequest = JSON.stringify(requests);
	console.log(rpcRequest);
	var httpRequest = new XMLHttpRequest();
	httpRequest.open("POST", jsonRpcUrl, true);
	httpRequest.setRequestHeader("Content-type", "application/json");
	httpRequest.setRequestHeader("Content-length", rpcRequest.length);
	httpRequest.setRequestHeader("Connection", "close");
	httpRequest.onreadystatechange = function() {
		if (httpRequest.readyState === XMLHttpRequest.DONE) {
			if (httpRequest.status !== 200 || httpRequest.responseText === "")
			{
				var errorText = qsTr("Unable to initiate request to the live network. Please verify your ethereum node is up.") + qsTr(" Error status: ")  + httpRequest.status;
				console.log(errorText);
				if (error)
					error(errorText);
			}
			else
			{
				console.log(httpRequest.responseText);
				callBack(httpRequest.status, httpRequest.responseText)
			}
		}
	}
	httpRequest.send(rpcRequest);
}

function contractFromToken(token)
{
	if (token.indexOf('<') === 0)
		return token.replace("<", "").replace(">", "").split(" - ")[0];
	return token;
}

