import QtQuick 2.2
import QtQuick.Controls 1.1
import QtQuick.Layouts 1.1
import QtQuick.Window 2.0
import org.ethereum.qml.QEther 1.0
import "js/TransactionHelper.js" as TransactionHelper

Window {
	id: modalTransactionDialog
	modality: Qt.WindowModal
	width:640
	height:640
	visible: false

	property int transactionIndex
	property alias transactionParams: paramsModel;
	property alias gas: gasField.value;
	property alias gasPrice: gasPriceField.value;
	property alias transactionValue: valueField.value;
	property alias functionId: functionComboBox.currentText;
	property var itemParams;
	property bool isConstructorTransaction;
	property bool useTransactionDefaultValue: false
	property var qType;

	signal accepted;

	function open(index, item) {
		qType = [];
		rowFunction.visible = !useTransactionDefaultValue;
		rowValue.visible = !useTransactionDefaultValue;
		rowGas.visible = !useTransactionDefaultValue;
		rowGasPrice.visible = !useTransactionDefaultValue;

		transactionIndex = index;
		gasField.value = item.gas;
		gasPriceField.value = item.gasPrice;
		valueField.value = item.value;
		var functionId = item.functionId;
		isConstructorTransaction = item.executeConstructor;
		rowFunction.visible = !item.executeConstructor;

		itemParams = item.parameters !== undefined ? item.parameters : {};
		functionsModel.clear();
		var functionIndex = -1;
		var functions = codeModel.code.contract.functions;
		for (var f = 0; f < functions.length; f++) {
			functionsModel.append({ text: functions[f].name });
			if (functions[f].name === item.functionId)
				functionIndex = f;
		}

		if (functionIndex == -1 && functionsModel.count > 0)
			functionIndex = 0; //@todo suggest unused function

		functionComboBox.currentIndex = functionIndex;
		paramsModel.clear();
		if (!item.executeConstructor)
			loadParameters();
		else
		{
			var parameters = codeModel.code.contract.constructor.parameters;
			for (var p = 0; p < parameters.length; p++)
				loadParameter(parameters[p]);
		}
		visible = true;
		valueField.focus = true;
	}

	function loadParameter(parameter)
	{
		var type = parameter.type;
		var pname = parameter.name;
		var varComponent;

		if (type.indexOf("int") !== -1)
			varComponent = Qt.createComponent("qrc:/qml/QIntType.qml");
		else if (type.indexOf("real") !== -1)
			varComponent = Qt.createComponent("qrc:/qml/QRealType.qml");
		else if (type.indexOf("string") !== -1 || type.indexOf("text") !== -1)
			varComponent = Qt.createComponent("qrc:/qml/QStringType.qml");
		else if (type.indexOf("hash") !== -1 || type.indexOf("address") !== -1)
			varComponent = Qt.createComponent("qrc:/qml/QHashType.qml");
		else if (type.indexOf("bool") !== -1)
			varComponent = Qt.createComponent("qrc:/qml/QBoolType.qml");

		var param = varComponent.createObject(modalTransactionDialog);
		var value = itemParams[pname] !== undefined ? itemParams[pname] : "";

		param.setValue(value);
		param.setDeclaration(parameter);
		qType.push({ name: pname, value: param });
		paramsModel.append({ name: pname, type: type, value: value });
	}

	function loadParameters() {
		paramsModel.clear();
		if (!paramsModel)
			return;
		if (functionComboBox.currentIndex >= 0 && functionComboBox.currentIndex < functionsModel.count) {
			var func = codeModel.code.contract.functions[functionComboBox.currentIndex];
			var parameters = func.parameters;
			for (var p = 0; p < parameters.length; p++)
				loadParameter(parameters[p]);
		}
	}

	function close()
	{
		visible = false;
	}

	function qTypeParam(name)
	{
		for (var k in qType)
		{
			if (qType[k].name === name)
				return qType[k].value;
		}
	}

	function getItem()
	{
		var item;
		if (!useTransactionDefaultValue)
		{
			item = {
				functionId: transactionDialog.functionId,
				gas: transactionDialog.gas,
				gasPrice: transactionDialog.gasPrice,
				value: transactionDialog.transactionValue,
				parameters: {},
				executeConstructor: isConstructorTransaction
			};
		}
		else
		{
			item = TransactionHelper.defaultTransaction();
			item.functionId = transactionDialog.functionId;
			item.executeConstructor = isConstructorTransaction;
		}

		if (isConstructorTransaction)
			item.functionId = qsTr("Constructor");

		var orderedQType = [];
		for (var p = 0; p < transactionDialog.transactionParams.count; p++) {
			var parameter = transactionDialog.transactionParams.get(p);
			var qtypeParam = qTypeParam(parameter.name);
			qtypeParam.setValue(parameter.value);
			orderedQType.push(qtypeParam);
			item.parameters[parameter.name] = parameter.value;
		}
		item.qType = orderedQType;
		return item;
	}

	ColumnLayout {
		id: dialogContent
		width: parent.width
		anchors.left: parent.left
		anchors.right: parent.right
		anchors.margins: 10
		spacing: 30
		RowLayout
		{
			id: rowFunction
			Layout.fillWidth: true
			height: 150
			Label {
				Layout.preferredWidth: 75
				text: qsTr("Function")
			}
			ComboBox {
				id: functionComboBox
				Layout.fillWidth: true
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


		RowLayout
		{
			id: rowValue
			Layout.fillWidth: true
			Label {
				Layout.preferredWidth: 75
				text: qsTr("Value")
			}
			Rectangle
			{
				Layout.fillWidth: true
				Ether {
					id: valueField
					edit: true
					displayFormattedValue: true
				}
			}
		}


		RowLayout
		{
			id: rowGas
			Layout.fillWidth: true
			Label {
				Layout.preferredWidth: 75
				text: qsTr("Gas")
			}
			Rectangle
			{
				Layout.fillWidth: true
				Ether {
					id: gasField
					edit: true
					displayFormattedValue: true
				}
			}
		}

		RowLayout
		{
			id: rowGasPrice
			Layout.fillWidth: true
			Label {
				Layout.preferredWidth: 75
				text: qsTr("Gas Price")
			}
			Rectangle
			{
				Layout.fillWidth: true
				Ether {
					id: gasPriceField
					edit: true
					displayFormattedValue: true
				}
			}
		}

		RowLayout
		{
			Layout.fillWidth: true
			Label {
				text: qsTr("Parameters")
				Layout.preferredWidth: 75
			}
			TableView {
				model: paramsModel
				Layout.preferredWidth: 120 * 2 + 240
				Layout.minimumHeight: 150
				Layout.preferredHeight: 400
				Layout.maximumHeight: 600
				TableViewColumn {
					role: "name"
					title: qsTr("Name")
					width: 120
				}
				TableViewColumn {
					role: "type"
					title: qsTr("Type")
					width: 120
				}
				TableViewColumn {
					role: "value"
					title: qsTr("Value")
					width: 240
				}

				rowDelegate: rowDelegate
				itemDelegate: editableDelegate
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


	ListModel {
		id: paramsModel
	}

	Component {
		id: rowDelegate
		Item {
			height: 100
		}
	}

	Component {
		id: editableDelegate
		Item {
			Loader {
				id: loaderEditor
				anchors.fill: parent
				anchors.margins: 4
				Connections {
					target: loaderEditor.item
					onTextChanged: {
						if (styleData.role === "value" && styleData.row < paramsModel.count)
							loaderEditor.updateValue(styleData.row, styleData.role, loaderEditor.item.text);
					}
				}

				function updateValue(row, role, value)
				{
					paramsModel.setProperty(styleData.row, styleData.role, value);
				}

				sourceComponent:
				{
					if (styleData.role === "value")
					{
						if (paramsModel.get(styleData.row) === undefined)
							return null;
						if (paramsModel.get(styleData.row).type.indexOf("int") !== -1)
							return intViewComp;
						else if (paramsModel.get(styleData.row).type.indexOf("bool") !== -1)
							return boolViewComp;
						else if (paramsModel.get(styleData.row).type.indexOf("string") !== -1)
							return stringViewComp;
						else if (paramsModel.get(styleData.row).type.indexOf("hash") !== -1)
							return hashViewComp;
					}
					else
						return editor;
				}

				Component
				{
					id: intViewComp
					QIntTypeView
					{
						id: intView
						text: styleData.value
					}
				}

				Component
				{
					id: boolViewComp
					QBoolTypeView
					{
						id: boolView
						defaultValue: "1"
						Component.onCompleted:
						{
							loaderEditor.updateValue(styleData.row, styleData.role,
													 (paramsModel.get(styleData.row).value === "" ? defaultValue :
																									paramsModel.get(styleData.row).value));
							text = (paramsModel.get(styleData.row).value === "" ? defaultValue : paramsModel.get(styleData.row).value);
						}
					}
				}

				Component
				{
					id: stringViewComp
					QStringTypeView
					{
						id: stringView
						text: styleData.value
					}
				}


				Component
				{
					id: hashViewComp
					QHashTypeView
					{
						id: hashView
						text: styleData.value
					}
				}

				Component {
					id: editor
					TextInput {
						id: textinput
						readOnly: true
						color: styleData.textColor
						text: styleData.value
						MouseArea {
							id: mouseArea
							anchors.fill: parent
							hoverEnabled: true
							onClicked: textinput.forceActiveFocus()
						}
					}
				}
			}
		}
	}
}
