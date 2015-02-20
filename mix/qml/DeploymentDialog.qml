import QtQuick 2.2
import QtQuick.Controls 1.1
import QtQuick.Layouts 1.1
import QtQuick.Window 2.0
import QtQuick.Controls.Styles 1.3
import org.ethereum.qml.QEther 1.0
import "js/TransactionHelper.js" as TransactionHelper
import "js/ProjectModel.js" as ProjectModelCode
import "js/QEtherHelper.js" as QEtherHelper
import "."


Window {
	id: modalDeploymentDialog
	modality: Qt.ApplicationModal
	width: 520
	height: 200
	visible: false
	property alias applicationUrlEth: applicationUrlEth.text
	property alias applicationUrlHttp: applicationUrlHttp.text
	property string root: "42f6279a5b6d350e1ce2a9ebef05657c79275c6a";
	property string eth: "31f6aee7f26e9d3320753c112ed34bcfc3c989b8";
	property string wallet: "c4040ef9635e7503bbbc74b73a9385ac78733d09";
	property string urlHintContract: "29a2e6d3c56ef7713a4e7229c3d1a23406f0161a"


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

	GridLayout
	{
		columns: 2
		anchors.top: parent.top
		anchors.left: parent.left
		anchors.topMargin: 10
		anchors.leftMargin: 10
		anchors.rightMargin: 10
		DefaultLabel
		{
			text: qsTr("Eth URL: ")
		}

		DefaultTextField
		{
			id: applicationUrlEth
		}

		DefaultLabel
		{
			text: qsTr("Http URL: ")
		}

		DefaultTextField
		{
			id: applicationUrlHttp
		}
	}

	RowLayout
	{
		anchors.bottom: parent.bottom
		anchors.right: parent.right;
		anchors.bottomMargin: 10
		Button {
			text: qsTr("Deploy");
			enabled: applicationUrlHttp.text !== ""
			onClicked: {
				if (Object.keys(projectModel.deploymentAddresses).length > 0)
					deployWarningDialog.open();
				else
					ProjectModelCode.startDeployProject();
			}
		}

		Button {
			text: qsTr("Rebuild Package");
			enabled: Object.keys(projectModel.deploymentAddresses).length > 0 && applicationUrlHttp.text !== ""
			onClicked: {
				var date = new Date();
				var deploymentId = date.toLocaleString(Qt.locale(), "ddMMyyHHmmsszzz");
				ProjectModelCode.finalizeDeployment(deploymentId, projectModel.deploymentAddresses);
			}
		}

		Button {
			text: qsTr("Close");
			onClicked: close();
		}

		Button {
			text: qsTr("Check Ownership");
			onClicked: {
				var requests = [];
				var ethStr = QEtherHelper.createString("eth");

				var ethHash = QEtherHelper.createHash(eth);

				requests.push({ //owner
					jsonrpc: "2.0",
					method: "eth_call",
					params: [ { "to": '0x' + modalDeploymentDialog.root, "data": "0xec7b9200" + ethStr.encodeValueAsString() } ],
					id: 3
				});

				requests.push({ //register
					jsonrpc: "2.0",
					method: "eth_call",
					params: [ { "to":  '0x' + modalDeploymentDialog.root, "data": "0x6be16bed" + ethStr.encodeValueAsString() } ],
					id: 4
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


		Button {
			text: qsTr("Generate registrar init");
			onClicked: {
				console.log("registering eth/wallet")
				var jsonRpcRequestId = 0;

				var requests = [];
				var ethStr = QEtherHelper.createString("eth");
				var ethHash = QEtherHelper.createHash(eth);
				requests.push({ //reserve
					jsonrpc: "2.0",
					method: "eth_transact",
					params: [ { "to": '0x' + modalDeploymentDialog.root, "data": "0x1c83171b" + ethStr.encodeValueAsString() } ],
					id: jsonRpcRequestId++
				});

				console.log("0x7d2e3ce9" + ethStr.encodeValueAsString() + pad(eth));
				console.log(ethStr.encodeValueAsString());
				console.log(pad(eth));

				requests.push({ //setRegister
					jsonrpc: "2.0",
					method: "eth_transact",
					params: [ { "to": '0x' + modalDeploymentDialog.root, "data": "0x96077307" + ethStr.encodeValueAsString() + pad(eth) /*ethHash.encodeValueAsString()*/ } ],
					id: jsonRpcRequestId++
				});

				var walletStr = QEtherHelper.createString("wallet");
				var walletHash = QEtherHelper.createHash(wallet);

				requests.push({ //reserve
					jsonrpc: "2.0",
					method: "eth_transact",
					params: [ { "to": '0x' + modalDeploymentDialog.eth, "data": "0x1c83171b" + walletStr.encodeValueAsString() } ],
					id: jsonRpcRequestId++
				});


				requests.push({ //setRegister
					jsonrpc: "2.0",
					method: "eth_transact",
					params: [ { "to": '0x' + modalDeploymentDialog.eth, "data": "0x96077307" + walletStr.encodeValueAsString() + pad(wallet) } ],
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
