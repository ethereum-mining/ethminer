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
	height:480
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

	signal accepted;

	function open(index, item) {
		valueLabel.visible = !useTransactionDefaultValue;
		valueField.visible = !useTransactionDefaultValue;
		gasLabel.visible = !useTransactionDefaultValue;
		gasFieldRect.visible = !useTransactionDefaultValue;
		gasPriceLabel.visible = !useTransactionDefaultValue;
		gasPriceRect.visible = !useTransactionDefaultValue;

		transactionIndex = index;
		gasField.value = item.gas;
		gasPriceField.value = item.gasPrice;
		valueField.value = item.value;
		var functionId = item.functionId;
		isConstructorTransaction = item.executeConstructor;
		functionLabel.visible = !item.executeConstructor;
		functionComboBox.visible = !item.executeConstructor;
		console.log(item.executeConstructor);

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
			functionIndex = 0; //@todo suggest unused funtion

		functionComboBox.currentIndex = functionIndex;
		paramsModel.clear();
		if (!item.executeConstructor)
			loadParameters();
		else
		{
			console.log("load ctro paramters");
			var parameters = codeModel.code.contract.constructor.parameters;
			for (var p = 0; p < parameters.length; p++) {
				var pname = parameters[p].name;
				paramsModel.append({ name: pname, type: parameters[p].type, value: itemParams[pname] !== undefined ? itemParams[pname].value() : "" });
			}
		}
		visible = true;
		valueField.focus = true;
	}

	function loadParameters() {
		if (!paramsModel)
			return;
		if (functionComboBox.currentIndex >= 0 && functionComboBox.currentIndex < functionsModel.count) {
			var func = codeModel.code.contract.functions[functionComboBox.currentIndex];
			var parameters = func.parameters;
			for (var p = 0; p < parameters.length; p++) {
				var pname = parameters[p].name;
				paramsModel.append({ name: pname, type: parameters[p].type, value: itemParams[pname] !== undefined ? itemParams[pname].value() : "" });
			}
		}
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

		for (var p = 0; p < transactionDialog.transactionParams.count; p++) {
			var parameter = transactionDialog.transactionParams.get(p);
			var intComponent = Qt.createComponent("qrc:/qml/BigIntValue.qml");
			var param = intComponent.createObject(modalTransactionDialog);
			console.log(param);
			console.log(JSON.stringify(param));
			console.log(item);
			console.log(JSON.stringify(item));

			param.setValue(parameter.value);
			item.parameters[parameter.name] = param;
		}
		return item;
	}

	GridLayout {
		id: dialogContent
		columns: 2
		anchors.fill: parent
		anchors.margins: 10
		rowSpacing: 10
		columnSpacing: 10

		Label {
			id: functionLabel;
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

		Label {
			id: valueLabel
			text: qsTr("Value")
		}
		Rectangle
		{
			id: valueFieldRect
			Layout.fillWidth: true
			Ether {
				id: valueField
				edit: true
				displayFormattedValue: true
			}
		}

		Label {
			id: gasLabel
			text: qsTr("Gas")
		}
		Rectangle
		{
			id: gasFieldRect
			Layout.fillWidth: true
			Ether {
				id: gasField
				edit: true
				displayFormattedValue: true
			}
		}

		Label {
			id: gasPriceLabel
			text: qsTr("Gas price")
		}
		Rectangle
		{
			id: gasPriceRect
			Layout.fillWidth: true
			Ether {
				id: gasPriceField
				edit: true
				displayFormattedValue: true
			}
		}

		Label {
			text: qsTr("Parameters")
		}
		TableView {
			model: paramsModel
			Layout.fillWidth: true

			TableViewColumn {
				role: "name"
				title: "Name"
				width: 120
			}
			TableViewColumn {
				role: "type"
				title: "Type"
				width: 120
			}
			TableViewColumn {
				role: "value"
				title: "Value"
				width: 120
			}

			itemDelegate: {
				return editableDelegate;
			}
		}
	}

	RowLayout
	{
		anchors.bottom: parent.bottom
		anchors.right: parent.right;

		Button {
			text: qsTr("Ok");
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
		id: editableDelegate
		Item {

			Text {
				width: parent.width
				anchors.margins: 4
				anchors.left: parent.left
				anchors.verticalCenter: parent.verticalCenter
				elide: styleData.elideMode
				text: styleData.value !== undefined ? styleData.value : ""
				color: styleData.textColor
				visible: !styleData.selected
			}
			Loader {
				id: loaderEditor
				anchors.fill: parent
				anchors.margins: 4
				Connections {
					target: loaderEditor.item
					onTextChanged: {
						if (styleData.role === "value" && styleData.row < paramsModel.count)
							paramsModel.setProperty(styleData.row, styleData.role, loaderEditor.item.text);
					}
				}
				sourceComponent: (styleData.selected) ? editor : null
				Component {
					id: editor
					TextInput {
						id: textinput
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
