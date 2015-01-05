import QtQuick 2.2
import QtQuick.Controls 1.2
import QtQuick.Layouts 1.1
import QtQuick.Window 2.0

Window {
	modality: Qt.WindowModal

	width:640
	height:480

	visible: false

	function open()
	{
		visible = true;
	}
	function close()
	{
		visible = false;
	}

	property alias focus : titleField.focus
	property alias transactionTitle : titleField.text
	property int transactionIndex
	property alias transactionParams : paramsModel;
	property alias gas : gasField.text;
	property alias gasPrice : gasPriceField.text;
	property alias transactionValue : valueField.text;
	property alias functionId : functionComboBox.currentText;
	property var model;

	signal accepted;

	function reset(index, m) {
		model = m;
		var item = model.getItem(index);
		transactionIndex = index;
		transactionTitle = item.title;
		gas = item.gas;
		gasPrice = item.gasPrice;
		transactionValue = item.value;
		var functionId = item.functionId;
		functionsModel.clear();
		var functionIndex = -1;
		var functions = model.getFunctions();
		for (var f = 0; f < functions.length; f++) {
			functionsModel.append({ text: functions[f] });
			if (functions[f] === item.functionId)
				functionIndex = f;
		}
		functionComboBox.currentIndex = functionIndex;
	}

	function loadParameters() {
		if (!paramsModel)
			return;
		paramsModel.clear();
		if (functionComboBox.currentIndex >= 0 && functionComboBox.currentIndex < functionsModel.count) {
			var parameters = model.getParameters(transactionIndex, functionsModel.get(functionComboBox.currentIndex).text);
			for (var p = 0; p < parameters.length; p++) {
				paramsModel.append({ name: parameters[p].name, type: parameters[p].type, value: parameters[p].value });
			}
		}
	}

	GridLayout {
		id: dialogContent
		columns: 2
		anchors.fill: parent
		anchors.margins: 10
		rowSpacing: 10
		columnSpacing: 10

		Label {
			text: qsTr("Title")
		}
		TextField {
			id: titleField
			focus: true
			Layout.fillWidth: true
		}

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
