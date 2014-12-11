import QtQuick 2.3
import QtQuick.Controls 1.2
import QtQuick.Dialogs 1.2
import QtQuick.Layouts 1.1
import QtQuick.Window 2.0

Dialog {
	modality: Qt.WindowModal
	standardButtons: StandardButton.Ok | StandardButton.Cancel

	width:640
	height:480

	property alias focus : titleField.focus
	property alias transactionTitle : titleField.text
	property int transactionId
	property int transactionParams;

	function reset(id, model) {
		var item = model.getItem(id);
		transactionId = id;
		transactionTitle = item.title;
	}

	GridLayout {
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
		TextField {
			id: functionField
			Layout.fillWidth: true
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
	ListModel {
		id: paramsModel
		Component.onCompleted: {
			for (var i=0 ; i < 3 ; ++i)
				paramsModel.append({"name":"Param " + i , "Type": "int", "value": i})
		}
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
					onAccepted: {
						//if (typeof styleData.value === 'number')
						//    paramsModel.setProperty(styleData.row, styleData.role, Number(parseFloat(loaderEditor.item.text).toFixed(0)))
						//else
						//    paramsModel.setProperty(styleData.row, styleData.role, loaderEditor.item.text)
					}
				}
				sourceComponent: styleData.selected ? editor : null
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
