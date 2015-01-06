import QtQuick 2.2
import QtQuick.Controls 1.2
import QtQuick.Layouts 1.1
import QtQuick.Window 2.0

Window {
	modality: Qt.WindowModal
	width:640
	height:480
	visible: false

	property int transactionIndex
	property alias transactionParams : paramsModel;
	property alias gas : gasField.text;
	property alias gasPrice : gasPriceField.text;
	property alias transactionValue : valueField.text;
	property alias functionId : functionComboBox.currentText;
	property var itemParams;

	signal accepted;

	function open(index, item) {
		transactionIndex = index;
		gas = item.gas;
		gasPrice = item.gasPrice;
		transactionValue = item.value;
		var functionId = item.functionId;
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
		loadParameters();
		visible = true;
		valueField.focus = true;
	}

	function loadParameters() {
		if (!paramsModel)
			return;
		paramsModel.clear();
		if (functionComboBox.currentIndex >= 0 && functionComboBox.currentIndex < functionsModel.count) {
			var func = codeModel.code.contract.functions[functionComboBox.currentIndex];
			var parameters = func.parameters;
			for (var p = 0; p < parameters.length; p++) {
				var pname = parameters[p].name;
				paramsModel.append({ name: pname, type: parameters[p].type, value: itemParams[pname] !== undefined ? itemParams[pname] : "" });
			}
		}
	}

	function close()
	{
		visible = false;
	}

	function getItem()
	{
		var item = {
			functionId: transactionDialog.functionId,
			gas: transactionDialog.gas,
			gasPrice: transactionDialog.gasPrice,
			value: transactionDialog.transactionValue,
			parameters: {}
		}
		for (var p = 0; p < transactionDialog.transactionParams.count; p++) {
			var parameter = transactionDialog.transactionParams.get(p);
			item.parameters[parameter.name] = parameter.value;
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
			text: qsTr("Value")
		}
		TextField {
			id: valueField
			Layout.fillWidth: true
		}

		Label {
			text: qsTr("Gas")
		}
		TextField {
			id: gasField
			Layout.fillWidth: true
		}

		Label {
			text: qsTr("Gas price")
		}
		TextField {
			id: gasPriceField
			Layout.fillWidth: true
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
