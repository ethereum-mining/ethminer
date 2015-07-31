import QtQuick 2.2
import QtQuick.Controls 1.1
import QtQuick.Layouts 1.1
import QtQuick.Dialogs 1.2
import QtQuick.Window 2.0
import QtQuick.Controls.Styles 1.3
import org.ethereum.qml.QEther 1.0
import "js/TransactionHelper.js" as TransactionHelper
import "js/InputValidator.js" as InputValidator
import "js/NetworkDeployment.js" as NetworkDeployment
import "js/QEtherHelper.js" as QEtherHelper
import "."

Dialog {
	id: modalTransactionDialog
	modality: Qt.ApplicationModal
	width: 580
	height: 500
	visible: false
	title:  editMode ? qsTr("Edit Transaction") : qsTr("Add Transaction")
	property bool editMode
	property int transactionIndex
	property int blockIndex
	property alias gas: gasValueEdit.gasValue;
	property alias gasAuto: gasAutoCheck.checked;
	property alias gasPrice: gasPriceField.value;
	property alias transactionValue: valueField.value;
	property string contractId: contractCreationComboBox.currentValue();
	property alias functionId: functionComboBox.currentText;
	property var paramValues;
	property var paramsModel: [];
	property bool useTransactionDefaultValue: false
	property alias stateAccounts: senderComboBox.model
	property bool saveStatus
	signal accepted;
	property int rowWidth: 500
	StateDialogStyle {
		id: transactionDialogStyle
	}

	function open(index, blockIdx, item) {
		transactionIndex = index
		blockIndex = blockIdx
		paramScroll.transactionIndex = index
		paramScroll.blockIndex = blockIdx
		saveStatus = item.saveStatus
		gasValueEdit.gasValue = item.gas;
		gasAutoCheck.checked = item.gasAuto ? true : false;
		gasPriceField.value = item.gasPrice;
		valueField.value = item.value;
		var contractId = item.contractId;
		var functionId = item.functionId;

		paramValues = item.parameters !== undefined ? item.parameters : {};
		if (item.sender)
			senderComboBox.select(item.sender);

		trTypeCreate.checked = item.isContractCreation
		trTypeSend.checked = !item.isFunctionCall
		trTypeExecute.checked = item.isFunctionCall && !item.isContractCreation

		load(item.isContractCreation, item.isFunctionCall, functionId, contractId)

		estimatedGas.updateView()
		visible = true;
	}

	function loadCtorParameters(contractId)
	{
		paramsModel = [];
		var contract = codeModel.contracts[contractId];
		if (contract) {
			var params = contract.contract.constructor.parameters;
			for (var p = 0; p < params.length; p++)
				loadParameter(params[p]);
		}
		initTypeLoader();
	}

	function loadFunctions(contractId)
	{
		functionsModel.clear();
		var contract = codeModel.contracts[contractId];
		if (contract) {
			var functions = codeModel.contracts[contractId].contract.functions;
			for (var f = 0; f < functions.length; f++) {
				if (functions[f].name !== contractId)
					functionsModel.append({ text: functions[f].name });
			}
		}
	}

	function selectContract(contractName)
	{
		for (var k = 0; k < contractsModel.count; k++)
		{
			if (contractsModel.get(k).cid === contractName)
			{
				contractComboBox.currentIndex = k;
				break;
			}
		}
	}

	function selectFunction(functionId)
	{
		var functionIndex = -1;
		for (var f = 0; f < functionsModel.count; f++)
			if (functionsModel.get(f).text === functionId)
				functionIndex = f;

		if (functionIndex == -1 && functionsModel.count > 0)
			functionIndex = 0; //@todo suggest unused function

		functionComboBox.currentIndex = functionIndex;
	}

	function loadParameter(parameter)
	{
		var type = parameter.type;
		var pname = parameter.name;
		paramsModel.push({ name: pname, type: type });
	}

	function loadParameters() {
		paramsModel = []
		if (functionComboBox.currentIndex >= 0 && functionComboBox.currentIndex < functionsModel.count) {
			var contract = codeModel.contracts[TransactionHelper.contractFromToken(recipientsAccount.currentValue())];
			if (contract) {
				var func = getFunction(functionComboBox.currentText, contract);
				if (func) {
					var parameters = func.parameters;
					for (var p = 0; p < parameters.length; p++)
						loadParameter(parameters[p]);
				}
			}
		}
		initTypeLoader();
	}

	function getFunction(name, contract)
	{
		for (var k in contract.contract.functions)
		{
			if (contract.contract.functions[k].name === name)
			{
				return contract.contract.functions[k]
			}
		}
		return null
	}

	function initTypeLoader()
	{
		paramScroll.clear()
		paramScroll.value = paramValues;
		paramScroll.members = paramsModel;
		paramScroll.updateView()
	}

	function acceptAndClose()
	{
		close();
		accepted();
	}

	function close()
	{
		visible = false;
	}

	function getItem()
	{
		var item;
		if (!useTransactionDefaultValue)
		{
			item = {
				contractId: transactionDialog.contractId,
				functionId: transactionDialog.functionId,
				gas: transactionDialog.gas,
				gasAuto: transactionDialog.gasAuto,
				gasPrice: transactionDialog.gasPrice,
				value: transactionDialog.transactionValue,
				parameters: {},
			};
		}
		else
		{
			item = TransactionHelper.defaultTransaction();
			item.contractId = transactionDialog.contractId;
			item.functionId = transactionDialog.functionId;
		}

		item.isContractCreation = trTypeCreate.checked;
		if (item.isContractCreation)
			item.functionId = item.contractId;
		item.isFunctionCall = trTypeExecute.checked

		if (!item.isContractCreation)
		{
			item.contractId = recipientsAccount.currentValue();
			item.label = TransactionHelper.contractFromToken(item.contractId) + "." + item.functionId + "()";
			if (recipientsAccount.current().type === "address")
			{
				item.functionId = "";
				item.isFunctionCall = false;
			}
		}
		else
		{
			item.isFunctionCall = true
			item.functionId = item.contractId;
			item.label = item.contractId + "." + item.contractId + "()";
		}
		item.saveStatus = saveStatus
		item.sender = senderComboBox.model[senderComboBox.currentIndex].secret;
		item.parameters = paramValues;
		return item;
	}	

	function load(isContractCreation, isFunctionCall, functionId, contractId)
	{
		if (!isContractCreation)
		{
			contractCreationComboBox.visible = false
			recipientsAccount.visible = true
			recipientsAccount.accounts = senderComboBox.model;
			amountLabel.text = qsTr("Amount")
			if (!isFunctionCall)
				recipientsAccount.subType = "address"
			else
				recipientsAccount.subType = "contract";
			recipientsAccount.load();
			recipientsAccount.init();
			if (contractId)
				recipientsAccount.select(contractId);
			if (functionId)
				selectFunction(functionId);
			else
				functionComboBox.currentIndex = 0
			if (isFunctionCall)
			{
				labelRecipient.text = qsTr("Recipient Contract")
				functionRect.show()
				loadFunctions(TransactionHelper.contractFromToken(recipientsAccount.currentValue()))
				loadParameters();
				paramScroll.updateView()
			}
			else
			{
				paramsModel = []
				paramScroll.updateView()
				labelRecipient.text = qsTr("Recipient Account")
				functionRect.hide()
			}
		}
		else
		{
			//contract creation
			contractsModel.clear();
			var contractIndex = -1;
			var contracts = codeModel.contracts;
			for (var c in contracts) {
				contractsModel.append({ cid: c, text: contracts[c].contract.name });
				if (contracts[c].contract.name === contractId)
					contractIndex = contractsModel.count - 1;
			}

			if (contractIndex == -1 && contractsModel.count > 0)
				contractIndex = 0; //@todo suggest unused contract
			contractCreationComboBox.currentIndex = contractIndex;
			contractCreationComboBox.visible = true
			labelRecipient.text = qsTr("Contract")
			amountLabel.text = qsTr("Endownment")
			functionRect.hide()
			recipientsAccount.visible = false
			loadCtorParameters(contractCreationComboBox.currentValue());
			paramScroll.updateView()
		}
	}

	contentItem: Rectangle {
		id: containerRect
		color: transactionDialogStyle.generic.backgroundColor
		anchors.fill: parent
		ScrollView
		{
			anchors.top: parent.top
			anchors.fill: parent
			ColumnLayout {
				Layout.preferredWidth: rowWidth
				anchors.top: parent.top
				anchors.topMargin: 10
				anchors.left: parent.left
				width: 500
				anchors.leftMargin:
				{
					return (containerRect.width - 530) /2
				}

				RowLayout
				{
					Rectangle
					{
						Layout.preferredWidth: 150
						Label {
							anchors.right: parent.right
							anchors.verticalCenter: parent.verticalCenter
							text: qsTr("Sender Account")
						}
					}

					ComboBox {
						function select(secret)
						{
							for (var i in model)
								if (model[i].secret === secret)
								{
									currentIndex = i;
									break;
								}
						}
						Layout.preferredWidth: 350
						id: senderComboBox
						currentIndex: 0
						textRole: "name"
						editable: false
					}
				}

				RowLayout
				{
					Rectangle
					{
						Layout.preferredWidth: 150
						Layout.preferredHeight: 80
						color: "transparent"
						Label
						{
							anchors.verticalCenter: parent.verticalCenter
							anchors.top: parent.top
							anchors.right: parent.right
							text: qsTr("Type of Transaction")
						}
					}

					Column
					{
						Layout.preferredWidth: 350
						Layout.preferredHeight: 90
						ExclusiveGroup {
							id: rbbuttonList
							onCurrentChanged: {
								if (current)
								{
									if (current.objectName === "trTypeSend")
									{
										recipientsAccount.visible = true
										contractCreationComboBox.visible = false
										modalTransactionDialog.load(false, false)
									}
									else if (current.objectName === "trTypeCreate")
									{
										contractCreationComboBox.visible = true
										recipientsAccount.visible = false
										modalTransactionDialog.load(true, true)
									}
									else if (current.objectName === "trTypeExecute")
									{
										recipientsAccount.visible = true
										contractCreationComboBox.visible = false
										modalTransactionDialog.load(false, true)
									}
								}
							}
						}

						RadioButton {
							id: trTypeSend
							objectName: "trTypeSend"
							exclusiveGroup: rbbuttonList
							height: 30
							text: qsTr("Send ether to account")

						}

						RadioButton {
							id: trTypeCreate
							objectName: "trTypeCreate"
							exclusiveGroup: rbbuttonList
							height: 30
							text: qsTr("Create Contract")
						}

						RadioButton {
							id: trTypeExecute
							objectName: "trTypeExecute"
							exclusiveGroup: rbbuttonList
							height: 30
							text: qsTr("Transact with Contract")
						}
					}
				}

				RowLayout
				{
					Rectangle
					{
						Layout.preferredWidth: 150
						Label {
							id: labelRecipient
							anchors.verticalCenter: parent.verticalCenter
							anchors.right: parent.right
							text: qsTr("Recipient Account")
						}
					}

					QAddressView
					{
						id: recipientsAccount
						displayInput: false
						onIndexChanged:
						{
							if (rbbuttonList.current.objectName === "trTypeExecute")
								loadFunctions(TransactionHelper.contractFromToken(currentValue()))
						}
					}

					ComboBox {
						id: contractCreationComboBox
						function currentValue() {
							return (currentIndex >=0 && currentIndex < contractsModel.count) ? contractsModel.get(currentIndex).cid : "";
						}
						Layout.preferredWidth: 350
						currentIndex: -1
						textRole: "text"
						editable: false
						model: ListModel {
							id: contractsModel
						}
						onCurrentIndexChanged: {
							loadCtorParameters(currentValue());
						}
					}
				}

				RowLayout
				{
					Rectangle
					{
						Layout.preferredWidth: 150
						id: functionRect

						function hide()
						{
							parent.visible = false
							functionRect.visible = false
							functionComboBox.visible = false
						}

						function show()
						{
							parent.visible = true
							functionRect.visible = true
							functionComboBox.visible = true
						}

						Label {
							anchors.verticalCenter: parent.verticalCenter
							anchors.right: parent.right
							text: qsTr("Function")
						}
					}

					ComboBox {
						id: functionComboBox
						Layout.preferredWidth: 350
						currentIndex: -1
						textRole: "text"
						editable: false
						model: ListModel {
							id: functionsModel
						}
						onCurrentIndexChanged: {
							loadParameters();
						}
					}
				}

				StructView
				{
					id: paramScroll
					members: paramsModel
					accounts: senderComboBox.model
					context: "parameter"
					Layout.fillWidth: true
					function updateView()
					{
						paramScroll.visible = paramsModel.length > 0
						paramScroll.Layout.preferredHeight = paramScroll.colHeight
						if (paramsModel.length === 0)
							paramScroll.height = 0
					}
				}

				RowLayout
				{
					Rectangle
					{
						Layout.preferredWidth: 150
						Label {
							id: amountLabel
							anchors.verticalCenter: parent.verticalCenter
							anchors.right: parent.right
							text: qsTr("Amount")
						}
					}

					Ether {
						Layout.preferredWidth: 350
						id: valueField
						edit: true
						displayFormattedValue: true
						displayUnitSelection: true
					}
				}

				Rectangle
				{
					Layout.preferredHeight: 30
					Layout.fillWidth: true
					color: "transparent"
					Rectangle
					{
						color: "#cccccc"
						height: 1
						width: parent.width
						anchors.verticalCenter: parent.verticalCenter
					}
				}

				Rectangle
				{
					height: 20
					color: "transparent"
					Layout.preferredWidth: 500
					Rectangle
					{

						anchors.horizontalCenter: parent.horizontalCenter
						Label {
							text: qsTr("Transaction fees")
							anchors.horizontalCenter: parent.horizontalCenter
						}
					}

				}

				RowLayout
				{
					Layout.preferredHeight: 45
					Rectangle
					{
						Layout.preferredWidth: 150
						Label {
							anchors.verticalCenter: parent.verticalCenter
							anchors.right: parent.right
							text: qsTr("Gas")
						}
					}

					Row
					{
						Layout.preferredWidth: 350
						DefaultTextField
						{
							property variant gasValue
							onGasValueChanged: text = gasValue.value();
							onTextChanged: gasValue.setValue(text);
							implicitWidth: 200
							enabled: !gasAutoCheck.checked
							id: gasValueEdit;

							Label
							{
								id: estimatedGas
								anchors.top: parent.bottom
								text: ""
								Connections
								{
									target: functionComboBox
									onCurrentIndexChanged:
									{
										estimatedGas.displayGas(TransactionHelper.contractFromToken(recipientsAccount.currentValue()), functionComboBox.currentText)
									}
								}

								function displayGas(contractName, functionName)
								{
									var gasCost = codeModel.gasCostBy(contractName, functionName);
									if (gasCost && gasCost.length > 0)
									{
										var gas = codeModel.txGas + codeModel.callStipend + parseInt(gasCost[0].gas)
										estimatedGas.text = qsTr("Estimated cost: ") + gasCost[0].gas + " gas"
									}
								}

								function updateView()
								{
									if (rbbuttonList.current.objectName === "trTypeExecute")
										estimatedGas.displayGas(TransactionHelper.contractFromToken(recipientsAccount.currentValue()), functionComboBox.currentText)
									else if (rbbuttonList.current.objectName === "trTypeCreate")
									{
										var contractName = contractCreationComboBox.currentValue()
										estimatedGas.displayGas(contractName, contractName)
									}
									else if (rbbuttonList.current.objectName === "trTypeSend")
									{
										var gas = codeModel.txGas + codeModel.callStipend
										estimatedGas.text = qsTr("Estimated cost: ") + gas + " gas"
									}
								}

								Connections
								{
									target: rbbuttonList
									onCurrentChanged: {
										estimatedGas.updateView()
									}
								}
							}
						}

						CheckBox
						{
							id: gasAutoCheck
							checked: true
							text: qsTr("Auto");
						}
					}
				}

				RowLayout
				{
					Layout.preferredWidth: 500
					Layout.preferredHeight: 45
					Rectangle
					{
						Layout.preferredWidth: 150
						Label {
							id: gasPriceLabel
							anchors.verticalCenter: parent.verticalCenter
							anchors.right: parent.right
							text: qsTr("Gas Price")

							Label {
								id: gasPriceMarket
								anchors.top: gasPriceLabel.bottom
								anchors.topMargin: 10
								Component.onCompleted:
								{
									NetworkDeployment.gasPrice(function(result)
									{
										gasPriceMarket.text = qsTr("Current market: ") + " " + QEtherHelper.createEther(result, QEther.Wei).format()
									}, function (){});
								}
							}
						}
					}

					Ether {
						Layout.preferredWidth: 400
						id: gasPriceField
						edit: true
						displayFormattedValue: false
						displayUnitSelection: true
					}
				}


				RowLayout
				{

					Layout.preferredWidth: 500
					Row
					{
						width: parent.width
						anchors.right: parent.right
						Button {
							id: updateBtn
							text: qsTr("Cancel");
							onClicked: close();
						}

						Button {
							text: editMode ? qsTr("Update") : qsTr("Ok")
							onClicked: {
								var invalid = InputValidator.validate(paramsModel, paramValues);
								if (invalid.length === 0)
								{
									close();
									accepted();
								}
								else
								{
									errorDialog.text = qsTr("Some parameters are invalid:\n");
									for (var k in invalid)
										errorDialog.text += invalid[k].message + "\n";
									errorDialog.open();
								}
							}
						}
					}

					MessageDialog {
						id: errorDialog
						standardButtons: StandardButton.Ok
						icon: StandardIcon.Critical
					}
				}

				RowLayout
				{
					Layout.preferredHeight: 30
					anchors.bottom: parent.bottom
				}
			}
		}
	}
}
