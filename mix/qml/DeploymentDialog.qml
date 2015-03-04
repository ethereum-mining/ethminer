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
	width: 735
	height: 480
	maximumWidth: width
	minimumWidth: width
	maximumHeight: height
	minimumHeight: height
	visible: false
	property alias applicationUrlEth: applicationUrlEth.text
	property alias applicationUrlHttp: applicationUrlHttp.text
	property string urlHintContract: urlHintAddr.text
	property string packageHash
	property alias packageBase64: base64Value.text
	property string eth: registrarAddr.text
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

	Column
	{
		spacing: 5
		anchors.fill: parent
		anchors.margins: 10
		ColumnLayout
		{
			id: containerDeploy
			Layout.fillWidth: true
			Layout.preferredHeight: 500
			RowLayout
			{
				DefaultLabel
				{
					text: qsTr("DEPLOYING")
					font.italic: true
					font.underline: true
					Layout.preferredWidth: 356
				}

				Button
				{
					action: displayHelpAction
					iconSource: "qrc:/qml/img/help.png"
				}

				Action {
					id: displayHelpAction
					tooltip: qsTr("Help")
					onTriggered: {
						Qt.openUrlExternally("https://github.com/ethereum/wiki/wiki/Mix:-The-DApp-IDE#deployment-to-network")
					}
				}
			}

			GridLayout
			{
				columns: 2
				width: parent.width

				DefaultLabel
				{
					text: qsTr("Registrar address:")
				}

				DefaultTextField
				{
					Layout.preferredWidth: 350
					id: registrarAddr
				}

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
					text: qsTr("Amount of gas to use for each contract deployment: ")
				}

				DefaultTextField
				{
					text: "20000"
					Layout.preferredWidth: 350
					id: gasToUseInput
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
						width: 300
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
					text: qsTr("Package (Base64): ")
				}

				TextArea
				{
					Layout.preferredWidth: 350
					readOnly: true
					id: base64Value
					height: 60
					enabled: base64Value.text != ""
				}
			}

			Row
			{
				Button {
					text: qsTr("Deploy");
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
						{
							if (contractRedeploy.checked)
								deployWarningDialog.open();
							else
								ProjectModelCode.startDeployProject(false);
						}
					}
				}

				CheckBox
				{
					id: contractRedeploy
					enabled: Object.keys(projectModel.deploymentAddresses).length > 0
					checked: Object.keys(projectModel.deploymentAddresses).length == 0
					text: qsTr("Deploy Contract(s)")
					anchors.verticalCenter: parent.verticalCenter
				}
			}
		}

		Rectangle
		{
			width: parent.width
			height: 1
			color: "#5891d3"
		}

		ColumnLayout
		{
			id: containerRegister
			Layout.fillWidth: true
			Layout.preferredHeight: 500
			RowLayout
			{
				DefaultLabel
				{
					text: qsTr("REGISTERING")
					font.italic: true
					font.underline: true
					Layout.preferredWidth: 356
				}

				Button
				{
					action: displayHelpAction
					iconSource: "qrc:/qml/img/help.png"
				}

				Button
				{
					action: openFolderAction
					iconSource: "qrc:/qml/img/openedfolder.png"
				}

				Action {
					id: openFolderAction
					enabled: deploymentDialog.packageBase64 !== ""
					tooltip: qsTr("Open Package Folder")
					onTriggered: {
						fileIo.openFileBrowser(projectModel.deploymentDir);
					}
				}
			}

			GridLayout
			{
				columns: 2
				Layout.fillWidth: true

				DefaultLabel
				{
					Layout.preferredWidth: 355
					text: qsTr("Url hint address:")
				}

				DefaultTextField
				{
					Layout.preferredWidth: 350
					id: urlHintAddr
					enabled: rowRegister.isOkToRegister()
				}

				DefaultLabel
				{
					Layout.preferredWidth: 355
					text: qsTr("Web Application Ressources URL: ")
				}

				DefaultTextField
				{
					Layout.preferredWidth: 350
					id: applicationUrlHttp
					enabled: rowRegister.isOkToRegister()
				}
			}

			Rectangle
			{
				id: rowRegister
				Layout.fillWidth: true

				function isOkToRegister()
				{
					return Object.keys(projectModel.deploymentAddresses).length > 0 && deploymentDialog.packageHash !== "";
				}

				Button {
					text: qsTr("Register");
					tooltip: qsTr("Register hosted Web Application.")
					enabled: rowRegister.isOkToRegister()
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
					anchors.right: parent.right
					text: qsTr("Cancel");
					onClicked: close();
				}
			}
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
}
