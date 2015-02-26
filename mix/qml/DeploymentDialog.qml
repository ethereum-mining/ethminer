import QtQuick 2.2
import QtQuick.Controls 1.1
import QtQuick.Layouts 1.1
import QtQuick.Window 2.0
import QtQuick.Dialogs 1.1
import QtQuick.Controls.Styles 1.3
import org.ethereum.qml.QEther 1.0
import "js/TransactionHelper.js" as TransactionHelper
import "js/ProjectModel.js" as ProjectModelCode
import "js/QEtherHelper.js" as QEtherHelper
import "."


Window {

	id: modalDeploymentDialog
	modality: Qt.ApplicationModal
	width: 930
	height: 350
	visible: false
	property alias applicationUrlEth: applicationUrlEth.text
	property alias applicationUrlHttp: applicationUrlHttp.text
	property string urlHintContract: "c83d3e22645fb015d02043a744921cc2f828c64d"
	property string packageHash
	property alias packageBase64: base64Value.text
	property string eth: "4c3f7330690ed3657d3fa20fe5717b84010528ae";
	property string yanndappRegistrar: "29a2e6d3c56ef7713a4e7229c3d1a23406f0161a";
	property string currentAccount
	property alias gasToUse: gasToUseInput.text

	color: Style.generic.layout.backgroundColor

	function close()
	{
		visible = false;
	}

	function open()
	{
		modalDeploymentDialog.setX((Screen.width - width) / 2);
		modalDeploymentDialog.setY((Screen.height - height) / 2);
		visible = true;

		var requests = [{
							//accounts
							jsonrpc: "2.0",
							method: "eth_accounts",
							params: null,
							id: 0
						}];

		TransactionHelper.rpcCall(requests, function(arg1, arg2)
		{
			modelAccounts.clear();
			var ids = JSON.parse(arg2)[0].result;
			requests = [];
			for (var k in ids)
			{
				modelAccounts.append({ "id": ids[k] })
				requests.push({
								  //accounts
								  jsonrpc: "2.0",
								  method: "eth_balanceAt",
								  params: [ids[k]],
								  id: k
							  });
			}

			if (ids.length > 0)
				currentAccount = modelAccounts.get(0).id;

			TransactionHelper.rpcCall(requests, function (request, response){
				var balanceRet = JSON.parse(response);
				for (var k in balanceRet)
				{
					var ether = QEtherHelper.createEther(balanceRet[k].result, QEther.Wei);
					comboAccounts.balances.push(ether.format());
				}
				balance.text = comboAccounts.balances[0];
			});
		});
	}

	function pad(h)
	{
		// TODO move this to QHashType class
		while (h.length < 64)
		{
			h = '0' + h;
		}
		return h;
	}

	function waitForTrCountToIncrement(callBack)
	{
		poolLog.callBack = callBack;
		poolLog.k = -1;
		poolLog.elapsed = 0;
		poolLog.start();
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
							  method: "eth_countAt",
							  params: [ currentAccount ],
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

	Rectangle
	{
		anchors.fill : parent
		anchors.margins: 10
		color: Style.generic.layout.backgroundColor
		GridLayout
		{
			columns: 2
			anchors.top: parent.top
			anchors.left: parent.left
			width: parent.width
			DefaultLabel
			{
				text: qsTr("Account used to deploy:")
			}

			Rectangle
			{
				width: 300
				height: 25
				color: "transparent"
				ComboBox {
					id: comboAccounts
					property var balances: []
					onCurrentIndexChanged : {
						if (modelAccounts.count > 0)
						{
							currentAccount = modelAccounts.get(currentIndex).id;
							balance.text = balances[currentIndex];
						}
					}
					model: ListModel {
						id: modelAccounts
					}
				}

				DefaultLabel
				{
					anchors.verticalCenter: parent.verticalCenter
					anchors.left: comboAccounts.right
					anchors.leftMargin: 20
					id: balance;
				}
			}


			DefaultLabel
			{
				text: qsTr("Ethereum Application URL: ")
			}

			DefaultTextField
			{
				Layout.fillWidth: true
				id: applicationUrlEth
			}

			DefaultLabel
			{
				text: qsTr("Web Application Ressources URL: ")
			}

			DefaultTextField
			{
				Layout.fillWidth: true
				id: applicationUrlHttp
			}

			DefaultLabel
			{
				text: qsTr("Amount of gas to use for each contract deployment: ")
			}

			DefaultTextField
			{
				text: "20000"
				Layout.fillWidth: true
				id: gasToUseInput
			}

			DefaultLabel
			{
				text: qsTr("Package (Base64): ")
			}

			TextArea
			{
				Layout.fillWidth: true
				readOnly: true
				id: base64Value
				height: 60
				enabled: base64Value.text != ""
			}
		}

		MessageDialog {
			id: deployDialog
			standardButtons: StandardButton.Ok
			icon: StandardIcon.Warning
		}

		RowLayout
		{
			anchors.bottom: parent.bottom
			anchors.right: parent.right;
			anchors.bottomMargin: 10
			Button {
				text: qsTr("Deploy contract / Package resources");
				tooltip: qsTr("Deploy contract and package resources files.")
				onClicked: {
					deployWarningDialog.open();
				}
			}

			Button {
				text: qsTr("Package resources only");
				tooltip: qsTr("Package resources files.")
				enabled: Object.keys(projectModel.deploymentAddresses).length > 0
				onClicked: {
					ProjectModelCode.startDeployProject(false);
				}
			}

			Button {
				text: qsTr("Open Package Directory");
				enabled: projectModel.deploymentDir !== ""
				onClicked: {
					fileIo.openFileBrowser(projectModel.deploymentDir);
				}
			}

			Button {
				text: qsTr("Register Web Application");
				tooltip: qsTr("Register hosted Web Application.")
				onClicked: {
					if (applicationUrlHttp.text === "" || deploymentDialog.packageHash === "")
					{
						deployDialog.title = text;
						deployDialog.text = qsTr("Please provide the link where the resources are stored and ensure the package is aleary built using the deployment step. ")
						deployDialog.open();
					}
					else
						ProjectModelCode.registerToUrlHint();
				}
			}

			Button {
				text: qsTr("Close");
				onClicked: close();
			}

			Button {
				text: qsTr("Checking eth/yanndapp");
				visible : false
				onClicked: {
					var requests = [];
					var ethStr = QEtherHelper.createString("yanndapp");

					requests.push({ //owner
									  jsonrpc: "2.0",
									  method: "eth_call",
									  params: [ { "to": '0x' + modalDeploymentDialog.eth, "data": "0xec7b9200" + ethStr.encodeValueAsString() } ], //check for yanndappRegistrar in eth
									  id: 1
								  });

					requests.push({ //register
									  jsonrpc: "2.0",
									  method: "eth_call",
									  params: [ { "to":  '0x' + modalDeploymentDialog.eth, "data": "0x6be16bed" + ethStr.encodeValueAsString() } ], //getregister yanndappRegistrar in eth
									  id: 2
								  });


					requests.push({ //getOwner
									  jsonrpc: "2.0",
									  method: "eth_call",
									  params: [ { "to": '0x' + modalDeploymentDialog.yanndappRegistrar, "data": "0x893d20e8" } ], //check owner of this registrar
									  id: 3
								  });

					/*requests.push({ //register
									  jsonrpc: "2.0",
									  method: "eth_call",
									  params: [ { "to":  '0x' + modalDeploymentDialog.yanndappRegistrar, "data": "0x6be16bed" + ethStr.encodeValueAsString() } ],
									  id: 2
								  });*/



					var jsonRpcUrl = "http://localhost:8080";
					var rpcRequest = JSON.stringify(requests);
					var httpRequest = new XMLHttpRequest();
					httpRequest.open("POST", jsonRpcUrl, true);
					httpRequest.setRequestHeader("Content-type", "application/json");
					httpRequest.setRequestHeader("Content-length", rpcRequest.length);
					httpRequest.setRequestHeader("Connection", "close");
					httpRequest.onreadystatechange = function() {
						if (httpRequest.readyState === XMLHttpRequest.DONE) {
							if (httpRequest.status === 200) {
								console.log(httpRequest.responseText);
							} else {
								var errorText = qsTr("path registration failed ") + httpRequest.status;
								console.log(errorText);
							}
						}
					}
					httpRequest.send(rpcRequest);
				}
			}

			Button {
				text: qsTr("add contracts");
				visible : false
				onClicked: {
					var jsonRpcRequestId = 0;
					var requests = [];
					requests.push({
									  jsonrpc: "2.0",
									  method: "eth_transact",
									  params: [ { "gas": 20000, "code": "0x60056011565b600180601c6000396000f35b6008600081905550560000" } ],
									  id: jsonRpcRequestId++
								  });

					var jsonRpcUrl = "http://localhost:8080";
					var rpcRequest = JSON.stringify(requests);
					var httpRequest = new XMLHttpRequest();
					httpRequest.open("POST", jsonRpcUrl, true);
					httpRequest.setRequestHeader("Content-type", "application/json");
					httpRequest.setRequestHeader("Content-length", rpcRequest.length);
					httpRequest.setRequestHeader("Connection", "close");
					httpRequest.onreadystatechange = function() {
						if (httpRequest.readyState === XMLHttpRequest.DONE) {
							console.log(httpRequest.responseText);
							var requests = [];

							requests.push({
											  jsonrpc: "2.0",
											  method: "eth_transact",
											  params: [ { "gas": 20000, "code": "0x60056011565b600180601c6000396000f35b6009600081905550560000" } ],
											  id: jsonRpcRequestId++
										  });
							rpcRequest = JSON.stringify(requests);
							httpRequest = new XMLHttpRequest();
							httpRequest.open("POST", jsonRpcUrl, true);
							httpRequest.setRequestHeader("Content-type", "application/json");
							httpRequest.setRequestHeader("Content-length", rpcRequest.length);
							httpRequest.setRequestHeader("Connection", "close");
							httpRequest.onreadystatechange = function() {
								if (httpRequest.readyState === XMLHttpRequest.DONE) {
									console.log(httpRequest.responseText);
								}
							}
							httpRequest.send(rpcRequest);
						}
					}
					httpRequest.send(rpcRequest);
				}
			}


			Button {
				text: qsTr("Registering eth/yanndapp");
				visible: false
				onClicked: {
					console.log("registering eth/yanndapp")
					var jsonRpcRequestId = 0;
					var requests = [];
					var ydapp = QEtherHelper.createString("yanndapp");

					requests.push({ //reserve
									  jsonrpc: "2.0",
									  method: "eth_transact",
									  params: [ { "gas": 2000, "to": '0x' + modalDeploymentDialog.eth, "data": "0x1c83171b" + ydapp.encodeValueAsString() } ],
									  id: jsonRpcRequestId++
								  });

					requests.push({ //setRegister
									  jsonrpc: "2.0",
									  method: "eth_transact",
									  params: [ { "gas": 2000, "to": '0x' + modalDeploymentDialog.eth, "data": "0x96077307" + ydapp.encodeValueAsString() + modalDeploymentDialog.pad(modalDeploymentDialog.yanndappRegistrar) } ],
									  id: jsonRpcRequestId++
								  });

					var jsonRpcUrl = "http://localhost:8080";
					var rpcRequest = JSON.stringify(requests);
					var httpRequest = new XMLHttpRequest();
					httpRequest.open("POST", jsonRpcUrl, true);
					httpRequest.setRequestHeader("Content-type", "application/json");
					httpRequest.setRequestHeader("Content-length", rpcRequest.length);
					httpRequest.setRequestHeader("Connection", "close");
					httpRequest.onreadystatechange = function() {
						if (httpRequest.readyState === XMLHttpRequest.DONE) {
							if (httpRequest.status === 200) {
								console.log(httpRequest.responseText);
							} else {
								var errorText = qsTr("path registration failed ") + httpRequest.status;
								console.log(errorText);
							}
						}
					}
					httpRequest.send(rpcRequest);
				}
			}
		}
	}
}
