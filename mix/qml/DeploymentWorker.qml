import QtQuick 2.2
import QtQuick.Controls 1.1
import QtQuick.Layouts 1.1
import QtQuick.Window 2.0
import QtQuick.Dialogs 1.2
import QtQuick.Controls.Styles 1.3
import org.ethereum.qml.QEther 1.0
import org.ethereum.qml.CodeModel 1.0
import org.ethereum.qml.ClientModel 1.0
import "js/TransactionHelper.js" as TransactionHelper
import "js/NetworkDeployment.js" as NetworkDeploymentCode
import "js/QEtherHelper.js" as QEtherHelper
import "."

Item
{
	property string currentAccount
	property string gasPrice
	property alias gasPriceInt: gasPriceInt
	property variant balances: ({})
	property variant accounts: []
	signal gasPriceLoaded()

	function renewCtx()
	{
		var requests = [{
							//accounts
							jsonrpc: "2.0",
							method: "eth_accounts",
							params: null,
							id: 0
						}];

		TransactionHelper.rpcCall(requests, function(arg1, arg2)
		{
			accounts = []
			var ids = JSON.parse(arg2)[0].result;
			requests = [];
			for (var k in ids)
			{
				requests.push({
								  //accounts
								  jsonrpc: "2.0",
								  method: "eth_getBalance",
								  params: [ids[k], 'latest'],
								  id: k
							  });
				accounts.push({ "id": ids[k] })
			}

			TransactionHelper.rpcCall(requests, function (request, response){
				var balanceRet = JSON.parse(response);
				for (var k in balanceRet)
				{
					var ether = QEtherHelper.createEther(balanceRet[k].result, QEther.Wei);
					balances[accounts[k]] = ether
				}
			}, function(){});
		}, function(){});

		NetworkDeploymentCode.gasPrice(function(price) {
			gasPrice = price;
			gasPriceInt.setValue(price);
			console.log("fjdsfkjds hfkdsf " + price)
			gasPriceLoaded()
		}, function(){});
	}

	function stopForInputError(inError)
	{
		errorDialog.text = "";
		if (inError.length > 0)
		{
			errorDialog.text = qsTr("The length of a string cannot exceed 32 characters.\nPlease verify the following value(s):\n\n")
			for (var k in inError)
				errorDialog.text += inError[k] + "\n";
			errorDialog.open();
			return true;
		}
		return false;
	}

	function waitForTrCountToIncrement(callBack)
	{
		poolLog.callBack = callBack;
		poolLog.k = -1;
		poolLog.elapsed = 0;
		poolLog.start();
	}

	Component.onCompleted:
	{
		renewCtx()
	}

	BigIntValue
	{
		id: gasPriceInt
	}

	function estimateGas(scenario, callback)
	{
		if (!clientModelGasEstimation.running)
		{
			var ctr = projectModel.codeEditor.getContracts()
			for (var k in ctr)
			{
				codeModelGasEstimation.registerCodeChange(ctr[k].document.documentId, ctr[k].getText());
			}
			gasEstimationConnect.callback = callback
			clientModelGasEstimation.setupScenario(scenario)
		}
	}

	Connections
	{
		id: gasEstimationConnect
		target: clientModelGasEstimation
		property var callback
		onRunComplete: {
				callback(clientModelGasEstimation.gasCosts)
		}
	}

	CodeModel {
		id: codeModelGasEstimation
	}

	ClientModel {
		id: clientModelGasEstimation
		codeModel: codeModelGasEstimation
		Component.onCompleted:
		{
			init("/tmp/bcgas/")
		}
	}

	Timer
	{
		id: poolLog
		property var callBack
		property int k: -1
		property int elapsed
		interval: 500
		running: false
		repeat: true
		onTriggered: {
			elapsed += interval;
			var requests = [];
			var jsonRpcRequestId = 0;
			requests.push({
							  jsonrpc: "2.0",
							  method: "eth_getTransactionCount",
							  params: [ currentAccount, "pending" ],
							  id: jsonRpcRequestId++
						  });
			TransactionHelper.rpcCall(requests, function (httpRequest, response){
				response = response.replace(/,0+/, ''); // ==> result:27,00000000
				var count = JSON.parse(response)[0].result
				if (k < parseInt(count) && k > 0)
				{
					stop();
					callBack(1);
				}
				else if (elapsed > 25000)
				{
					stop();
					callBack(-1);
				}
				else
					k = parseInt(JSON.parse(response)[0].result);
			})
		}
	}	
}

