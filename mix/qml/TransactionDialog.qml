import QtQuick 2.2
import QtQuick.Controls 1.1
import QtQuick.Layouts 1.1
import QtQuick.Dialogs 1.2
import QtQuick.Window 2.0
import QtQuick.Controls.Styles 1.3
import org.ethereum.qml.QEther 1.0
import "js/TransactionHelper.js" as TransactionHelper
import "."

Dialog {
	id: modalTransactionDialog
	modality: Qt.ApplicationModal
	width: 570
	height: 500
	visible: false
	title: qsTr("Edit Transaction")
	property int transactionIndex
	property alias gas: gasValueEdit.gasValue;
	property alias gasAuto: gasAutoCheck.checked;
	property alias gasPrice: gasPriceField.value;
	property alias transactionValue: valueField.value;
	property string contractId: contractComboBox.currentValue();
	property alias functionId: functionComboBox.currentText;
	property var paramValues;
	property var paramsModel: [];
	property bool useTransactionDefaultValue: false
	property alias stateAccounts: senderComboBox.model
	signal accepted;

	StateDialogStyle {
		id: transactionDialogStyle
	}

	function open(index, item) {
		rowFunction.visible = !useTransactionDefaultValue;
		rowValue.visible = !useTransactionDefaultValue;
		rowGas.visible = !useTransactionDefaultValue;
		rowGasPrice.visible = !useTransactionDefaultValue;

		transactionIndex = index;
		typeLoader.transactionIndex = index;

		gasValueEdit.gasValue = item.gas;
		gasAutoCheck.checked = item.gasAuto ? true : false;
		gasPriceField.value = item.gasPrice;
		valueField.value = item.value;
		var contractId = item.contractId;
		var functionId = item.functionId;
		rowFunction.visible = true;

		paramValues = item.parameters !== undefined ? item.parameters : {};
		if (item.sender)
			senderComboBox.select(item.sender);

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
		contractComboBox.currentIndex = contractIndex;

		loadFunctions(contractComboBox.currentValue());
		selectFunction(functionId);

		paramsModel = [];
		if (functionId !== contractComboBox.currentValue())
			loadParameters();
		else {
			var contract = codeModel.contracts[contractId];
			if (contract) {
				var params = contract.contract.constructor.parameters;
				for (var p = 0; p < params.length; p++)
					loadParameter(params[p]);
			}
		}
		initTypeLoader();

		visible = true;
		valueField.focus = true;
	}

	function loadFunctions(contractId)
	{
		functionsModel.clear();
		var contract = codeModel.contracts[contractId];
		if (contract) {
			var functions = codeModel.contracts[contractId].contract.functions;
			for (var f = 0; f < functions.length; f++) {
				functionsModel.append({ text: functions[f].name });
			}
		}
		//append constructor
		functionsModel.append({ text: contractId });

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
			var contract = codeModel.contracts[contractComboBox.currentValue()];
			if (contract) {
				var func = contract.contract.functions[functionComboBox.currentIndex];
				if (func) {
					var parameters = func.parameters;
					for (var p = 0; p < parameters.length; p++)
						loadParameter(parameters[p]);
				}
			}
		}
		initTypeLoader();
	}

	function initTypeLoader()
	{
		typeLoader.value = {}
		typeLoader.members = []
		typeLoader.value = paramValues;
		typeLoader.members = paramsModel;
		paramLabel.visible = paramsModel.length > 0;
		paramScroll.visible = paramsModel.length > 0;
		modalTransactionDialog.height = (paramsModel.length > 0 ? 500 : 300);
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

		item.sender = senderComboBox.model[senderComboBox.currentIndex].secret;
		item.parameters = paramValues;
		return item;
	}
	contentItem: Rectangle {
		color: transactionDialogStyle.generic.backgroundColor
		ColumnLayout {
			anchors.fill: parent
			ColumnLayout {
				anchors.fill: parent
				anchors.margins: 10

				ColumnLayout {
					id: dialogContent
					anchors.top: parent.top
					spacing: 10
					RowLayout
					{
						id: rowSender
						Layout.fillWidth: true
						height: 150
						DefaultLabel {
							Layout.preferredWidth: 75
							text: qsTr("Sender")
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

							id: senderComboBox
							Layout.preferredWidth: 350
							currentIndex: 0
							textRole: "name"
							editable: false
						}
					}

					RowLayout
					{
						id: rowContract
						Layout.fillWidth: true
						height: 150
						DefaultLabel {
							Layout.preferredWidth: 75
							text: qsTr("Contract")
						}
						ComboBox {
							id: contractComboBox
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
								loadFunctions(currentValue());
							}
						}
					}

					RowLayout
					{
						id: rowFunction
						Layout.fillWidth: true
						height: 150
						DefaultLabel {
							Layout.preferredWidth: 75
							text: qsTr("Function")
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

					CommonSeparator
					{
						Layout.fillWidth: true
					}

					RowLayout
					{
						id: rowValue
						Layout.fillWidth: true
						height: 150
						DefaultLabel {
							Layout.preferredWidth: 75
							text: qsTr("Value")
						}
						Ether {
							id: valueField
							edit: true
							displayFormattedValue: true
						}
					}

					CommonSeparator
					{
						Layout.fillWidth: true
					}

					RowLayout
					{
						id: rowGas
						Layout.fillWidth: true
						height: 150
						DefaultLabel {
							Layout.preferredWidth: 75
							text: qsTr("Gas")
						}

						DefaultTextField
						{
							property variant gasValue
							onGasValueChanged: text = gasValue.value();
							onTextChanged: gasValue.setValue(text);
							implicitWidth: 200
							enabled: !gasAutoCheck.checked
							id: gasValueEdit;
						}

						CheckBox
						{
							id: gasAutoCheck
							checked: true
							text: qsTr("Auto");
						}
					}

					CommonSeparator
					{
						Layout.fillWidth: true
					}

					RowLayout
					{
						id: rowGasPrice
						Layout.fillWidth: true
						height: 150
						DefaultLabel {
							Layout.preferredWidth: 75
							text: qsTr("Gas Price")
						}
						Ether {
							id: gasPriceField
							edit: true
							displayFormattedValue: true
						}
					}

					CommonSeparator
					{
						Layout.fillWidth: true
					}

					DefaultLabel {
						id: paramLabel
						text: qsTr("Parameters:")
						Layout.preferredWidth: 75
					}

					ScrollView
					{
						id: paramScroll
						anchors.top: paramLabel.bottom
						anchors.topMargin: 10
						Layout.fillWidth: true
						Layout.fillHeight: true
						StructView
						{
							id: typeLoader
							Layout.preferredWidth: 150
							members: paramsModel;
							accounts: senderComboBox.model
							context: "parameter"
						}
					}

					CommonSeparator
					{
						Layout.fillWidth: true
						visible: paramsModel.length > 0
					}
				}
			}

			RowLayout
			{
				anchors.bottom: parent.bottom
				anchors.right: parent.right;

				Button {
					text: qsTr("OK");
					onClicked: {
						close();
						accepted();
					}
				}

				Button {
					text: qsTr("Cancel");
					onClicked: close();
				}
			}
		}
	}
}

