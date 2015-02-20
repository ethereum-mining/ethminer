import QtQuick 2.2
import QtQuick.Controls 1.1
import QtQuick.Controls.Styles 1.1
import QtQuick.Dialogs 1.1
import QtQuick.Layouts 1.1

Item {

	property bool showLogs: true
	property ListModel fullModel: ListModel{}
	property ListModel transactionModel: ListModel{}
	onShowLogsChanged: {
		logTable.model = showLogs ? fullModel : transactionModel
	}

	Action {
		id: addStateAction
		text: "Add State"
		shortcut: "Ctrl+Alt+T"
		enabled: codeModel.hasContract && !clientModel.running;
		onTriggered: projectModel.stateListModel.addState();
	}
	Action {
		id: editStateAction
		text: "Edit State"
		shortcut: "Ctrl+Alt+T"
		enabled: codeModel.hasContract && !clientModel.running && statesCombo.currentIndex >= 0 && projectModel.stateListModel.count > 0;
		onTriggered: projectModel.stateListModel.editState(statesCombo.currentIndex);
	}

	ColumnLayout {
		anchors.fill: parent
		RowLayout {

			Connections
			{
				target: projectModel
				onProjectSaved:
				{
					if (codeModel.hasContract && !clientModel.running)
						projectModel.stateListModel.debugDefaultState();
				}
			}

			ComboBox {
				id: statesCombo
				model: projectModel.stateListModel
				width: 150
				editable: false
				textRole: "title"
				onActivated:  {
					model.runState(index);
				}
				Connections {
					target: projectModel.stateListModel
					onStateRun: {
						if (statesCombo.currentIndex !== index)
							statesCombo.currentIndex = index;
					}
				}
			}
			Button
			{
				anchors.rightMargin: 9
				anchors.verticalCenter: parent.verticalCenter
				action: editStateAction
			}
			Button
			{
				anchors.rightMargin: 9
				anchors.verticalCenter: parent.verticalCenter
				action: addStateAction
			}
			Button
			{
				anchors.rightMargin: 9
				anchors.verticalCenter: parent.verticalCenter
				action: mineAction
			}

			CheckBox {
				id: recording
				text: qsTr("Record transactions");
				checked: true
				Layout.fillWidth: true


			}
		}
		TableView {
			id: logTable
			Layout.fillWidth: true
			Layout.fillHeight: true
			model: fullModel

			TableViewColumn {
				role: "transactionIndex"
				title: qsTr("Index")
				width: 40
			}
			TableViewColumn {
				role: "contract"
				title: qsTr("Contract")
				width: 100
			}
			TableViewColumn {
				role: "function"
				title: qsTr("Function")
				width: 120
			}
			TableViewColumn {
				role: "value"
				title: qsTr("Value")
				width: 60
			}
			TableViewColumn {
				role: "address"
				title: qsTr("Address")
				width: 120
			}
			TableViewColumn {
				role: "returned"
				title: qsTr("Returned")
				width: 120
			}
			onActivated:  {
				var item = logTable.model.get(row);
				clientModel.debugRecord(item.recordIndex);
			}
			Keys.onPressed: {
				if ((event.modifiers & Qt.ControlModifier) && event.key === Qt.Key_C && currentRow >=0 && currentRow < logTable.model.count) {
					var item = logTable.model.get(currentRow);
					appContext.toClipboard(item.returned);
				}
			}
		}
	}

	Connections {
		target: clientModel
		onStateCleared: {
			fullModel.clear();
			transactionModel.clear();
		}
		onNewRecord: {
			if (recording.checked)
			{
				fullModel.append(_r);
				if (!_r.call)
					transactionModel.append(_r);
			}
		}
	}

}
