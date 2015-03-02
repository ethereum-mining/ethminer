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
	property string urlHintContract: "c83d3e22645fb015d02043a744921cc2f828c64d" /* TODO: replace with the good address */
	property string packageHash
	property alias packageBase64: base64Value.text
	property string eth: "4c3f7330690ed3657d3fa20fe5717b84010528ae"; /* TODO: replace with the good address */
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

			Rectangle
			{
				Layout.fillWidth: true
				height: 25
				color: "transparent"
				DefaultTextField
				{
					width: 350
					id: applicationUrlEth
				}

				DefaultLabel
				{
					anchors.verticalCenter: parent.verticalCenter;
					anchors.left: applicationUrlEth.right
					text: "/" + projectModel.projectTitle
				}
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

		MessageDialog {
			id: errorDialog
			standardButtons: StandardButton.Ok
			icon: StandardIcon.Critical
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
					var inError = [];
					var ethUrl = ProjectModelCode.formatAppUrl(applicationUrlEth.text);
					for (var k in ethUrl)
					{
						if (ethUrl[k].length > 32)
							inError.push(qsTr("Member too long: " + ethUrl[k]) + "\n");
					}
					if (!stopForInputError(inError))
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
						deployDialog.text = qsTr("Please provide the link where the resources are stored and ensure the package is aleary built using the deployment step.")
						deployDialog.open();
						return;
					}
					var inError = [];
					if (applicationUrlHttp.text.length > 32)
						inError.push(qsTr(applicationUrlHttp.text));
					if (!stopForInputError(inError))
						ProjectModelCode.registerToUrlHint();
				}
			}

			Button {
				text: qsTr("Close");
				onClicked: close();
			}
		}
	}
}
