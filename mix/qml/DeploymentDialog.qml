import QtQuick 2.2
import QtQuick.Controls 1.1
import QtQuick.Layouts 1.1
import QtQuick.Window 2.0
import QtQuick.Dialogs 1.2
import QtQuick.Controls.Styles 1.3
import org.ethereum.qml.QEther 1.0
import "js/TransactionHelper.js" as TransactionHelper
import "js/NetworkDeployment.js" as NetworkDeploymentCode
import "js/QEtherHelper.js" as QEtherHelper
import "."


Dialog {
	id: modalDeploymentDialog
	modality: Qt.ApplicationModal
	width: 735
	height: 450
	visible: false
	property int ownedRegistrarDeployGas: 1179075 // TODO: Use sol library to calculate gas requirement for each tr.
	property int ownedRegistrarSetSubRegistrarGas: 50000
	property int ownedRegistrarSetContentHashGas: 50000
	property int urlHintSuggestUrlGas: 70000
	property alias applicationUrlEth: applicationUrlEth.text
	property alias applicationUrlHttp: applicationUrlHttp.text
	property alias localPackageUrl: localPackageUrl.text
	property string packageHash
	property string packageBase64
	property string eth: registrarAddr.text
	property string currentAccount
	property string gasPrice
	property variant paramsModel: []

	function close()
	{
		visible = false;
	}

	function open()
	{
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
								  method: "eth_getBalance",
								  params: [ids[k], 'latest'],
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
					comboAccounts.weiBalances.push(balanceRet[k].result);
				}
				balance.text = comboAccounts.balances[0];
			});
		});

		if (clientModel.gasCosts.length === 0)
		{
			errorDialog.text = qsTr("Please run the state one time before deploying in order to calculate gas requirement.");
			errorDialog.open();
		}
		else
		{
			NetworkDeploymentCode.gasPrice(function(price) {
				gasPrice = price;
				gasPriceInt.setValue(gasPrice);
				ctrDeployCtrLabel.calculateContractDeployGas();
				ctrRegisterLabel.calculateRegisterGas();
			});
		}
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

	BigIntValue
	{
		id: gasPriceInt
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
				console.log("count " + count);
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

	SourceSansProRegular
	{
		id: lightFont
	}

	contentItem: Rectangle {
		color: appStyle.generic.layout.backgroundColor
		anchors.fill: parent
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
					Rectangle
					{
						Layout.preferredWidth: 357
						DefaultLabel
						{
							text: qsTr("Deployment")
							font.family: lightFont.name
							font.underline: true
							anchors.centerIn: parent
						}
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

					Button
					{
						action: b64Action
						iconSource: "qrc:/qml/img/b64.png"
					}

					Action {
						id: b64Action
						enabled: deploymentDialog.packageBase64 !== ""
						tooltip: qsTr("Copy Base64 conversion to ClipBoard")
						onTriggered: {
							clipboard.text = deploymentDialog.packageBase64;
						}
					}

					Button
					{
						action: exitAction
						iconSource: "qrc:/qml/img/exit.png"
					}

					Action {
						id: exitAction
						tooltip: qsTr("Exit")
						onTriggered: {
							close()
						}
					}
				}

				GridLayout
				{
					columns: 2
					width: parent.width

					DefaultLabel
					{
						text: qsTr("State:")
					}

					Rectangle
					{
						width: 300
						color: "transparent"
						height: 25
						id: paramsRect
						ComboBox
						{
							id: statesList
							textRole: "title"
							model: projectModel.stateListModel
							onCurrentIndexChanged : {
								ctrDeployCtrLabel.calculateContractDeployGas();
								ctrRegisterLabel.calculateRegisterGas();
							}
						}
					}

					DefaultLabel
					{
						text: qsTr("Root Registrar address:")
						visible: true //still use it for now in dev env.
					}

					DefaultTextField
					{
						Layout.preferredWidth: 350
						id: registrarAddr
						text: "c6d9d2cd449a754c494264e1809c50e34d64562b"
						visible: true
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
							property var weiBalances: []
							onCurrentIndexChanged : {
								if (modelAccounts.count > 0)
								{
									currentAccount = modelAccounts.get(currentIndex).id;
									balance.text = balances[currentIndex];
									balanceInt.setValue(weiBalances[currentIndex]);
									ctrDeployCtrLabel.calculateContractDeployGas();
									ctrRegisterLabel.calculateRegisterGas();
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

						BigIntValue
						{
							id: balanceInt
						}
					}
				}

				DefaultLabel
				{
					text: qsTr("Amount of gas to use for contract deployment: ")
					id: ctrDeployCtrLabel
					function calculateContractDeployGas()
					{
						var ether = QEtherHelper.createBigInt(NetworkDeploymentCode.gasUsed());
						var gasTotal = ether.multiply(gasPriceInt);
						gasToUseInput.value = QEtherHelper.createEther(gasTotal.value(), QEther.Wei, parent);
						gasToUseDeployInput.update();
					}
				}

				Ether {
					id: gasToUseInput
					displayUnitSelection: false
					displayFormattedValue: true
					Layout.preferredWidth: 350
				}

				DefaultLabel
				{
					text: qsTr("Amount of gas to use for dapp registration: ")
					id: ctrRegisterLabel
					function calculateRegisterGas()
					{
						if (!modalDeploymentDialog.visible)
							return;
						appUrlFormatted.text = NetworkDeploymentCode.formatAppUrl(applicationUrlEth.text).join('/');
						NetworkDeploymentCode.checkPathCreationCost(function(pathCreationCost)
						{
							var ether = QEtherHelper.createBigInt(pathCreationCost);
							var gasTotal = ether.multiply(gasPriceInt);
							gasToUseDeployInput.value = QEtherHelper.createEther(gasTotal.value(), QEther.Wei, parent);
							gasToUseDeployInput.update();
						});
					}
				}

				Ether {
					id: gasToUseDeployInput
					displayUnitSelection: false
					displayFormattedValue: true
					Layout.preferredWidth: 350
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
						width: 200
						id: applicationUrlEth
						onTextChanged: {
							ctrRegisterLabel.calculateRegisterGas();
						}
					}

					DefaultLabel
					{
						id: appUrlFormatted
						anchors.verticalCenter: parent.verticalCenter;
						anchors.left: applicationUrlEth.right
						font.italic: true
						font.pointSize: appStyle.absoluteSize(-1)
					}
				}
			}

			RowLayout
			{
				Layout.fillWidth: true
				Rectangle
				{
					Layout.preferredWidth: 357
					color: "transparent"
				}

				Button
				{
					id: deployButton
					action: runAction
					iconSource: "qrc:/qml/img/run.png"
				}

				Action {
					id: runAction
					tooltip: qsTr("Deploy contract(s) and Package resources files.")
					onTriggered: {
						var inError = [];
						var ethUrl = NetworkDeploymentCode.formatAppUrl(applicationUrlEth.text);
						for (var k in ethUrl)
						{
							if (ethUrl[k].length > 32)
								inError.push(qsTr("Member too long: " + ethUrl[k]) + "\n");
						}
						if (!stopForInputError(inError))
						{
							projectModel.deployedState = statesList.currentText;
							if (contractRedeploy.checked)
								deployWarningDialog.open();
							else
								NetworkDeploymentCode.startDeployProject(false);
						}
					}
				}

				CheckBox
				{
					anchors.left: deployButton.right
					id: contractRedeploy
					enabled: Object.keys(projectModel.deploymentAddresses).length > 0
					checked: Object.keys(projectModel.deploymentAddresses).length == 0
					text: qsTr("Deploy Contract(s)")
					anchors.verticalCenter: parent.verticalCenter
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
					Layout.preferredHeight: 25
					Rectangle
					{
						Layout.preferredWidth: 356
						DefaultLabel
						{
							text: qsTr("Registration")
							font.family: lightFont.name
							font.underline: true
							anchors.centerIn: parent
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
						text: qsTr("Local package URL")
					}

					DefaultTextField
					{
						Layout.preferredWidth: 350
						id: localPackageUrl
						readOnly: true
					}

					DefaultLabel
					{
						Layout.preferredWidth: 355
						text: qsTr("Web Application Resources URL: ")
					}

					DefaultTextField
					{
						Layout.preferredWidth: 350
						id: applicationUrlHttp
						enabled: rowRegister.isOkToRegister()
					}
				}

				RowLayout
				{
					id: rowRegister
					Layout.fillWidth: true

					Rectangle
					{
						Layout.preferredWidth: 357
						color: "transparent"
					}

					function isOkToRegister()
					{
						return Object.keys(projectModel.deploymentAddresses).length > 0 && deploymentDialog.packageHash !== "";
					}

					Button {
						action: registerAction
						iconSource: "qrc:/qml/img/note.png"
					}

					BigIntValue
					{
						id: registerUrlHintGas
						Component.onCompleted:
						{
							setValue(modalDeploymentDialog.urlHintSuggestUrlGas);
						}
					}

					Action {
						id: registerAction
						enabled: rowRegister.isOkToRegister()
						tooltip: qsTr("Register hosted Web Application.")
						onTriggered: {
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
								NetworkDeploymentCode.registerToUrlHint();
						}
					}
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
