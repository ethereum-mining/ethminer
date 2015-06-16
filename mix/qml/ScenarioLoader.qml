import QtQuick 2.2
import QtQuick.Controls 1.1
import QtQuick.Controls.Styles 1.1
import QtQuick.Layouts 1.1
import QtQuick.Dialogs 1.2
import QtQuick.Window 2.0
import QtQuick.Dialogs 1.1
import Qt.labs.settings 1.0
import "js/Debugger.js" as Debugger
import "js/ErrorLocationFormater.js" as ErrorLocationFormater
import "."

RowLayout
{
	signal restored(variant scenario)
	signal saved(variant scenario)
	signal duplicated(variant scenario)
	signal loaded(variant scenario)
	spacing: 0
	function init()
	{
		scenarioList.load()
	}

	id: blockChainSelector

	Dialog {
		id: newStateWin
		modality: Qt.ApplicationModal
		title: qsTr("New Project");

		width: 320
		height: 120

		visible: false

		contentItem: Rectangle {
			anchors.fill: parent
			anchors.margins: 10
			RowLayout {
				anchors.verticalCenter: parent.verticalCenter
				Text {
					text: qsTr("Name:")
				}

				Rectangle
				{
					Layout.preferredWidth: 250
					Layout.preferredHeight: parent.height
					border.width: 1
					border.color: "#cccccc"
					TextInput
					{
						anchors.fill: parent
						id: stateName
					}
				}
			}
			RowLayout
			{
				anchors.bottom: parent.bottom
				anchors.right: parent.right;
				function acceptAndClose()
				{
					var item = projectModel.stateListModel.createDefaultState();
					item.title = stateName.text
					projectModel.stateListModel.appendState(item)
					projectModel.stateListModel.save()
					close()
					scenarioList.currentIndex = projectModel.stateListModel.count - 1
				}

				function close()
				{
					newStateWin.close()
					stateName.text = ""
				}

				Button {
					id: okButton;
					enabled: stateName.text !== ""
					text: qsTr("OK");
					onClicked: {
						parent.acceptAndClose();
					}
				}
				Button {
					text: qsTr("Cancel");
					onClicked: parent.close();
				}
			}
		}
	}

	ComboBox
	{
		id: scenarioList
		model: projectModel.stateListModel
		textRole: "title"
		onCurrentIndexChanged:
		{
			restoreScenario.restore()
		}

		function load()
		{
			var state = projectModel.stateListModel.getState(currentIndex)
			loaded(state)
		}
	}

	Row
	{
		Layout.preferredWidth: 100 * 4
		Layout.preferredHeight: 30
		spacing: 0
		ScenarioButton {
			id: restoreScenario
			width: 100
			height: 30
			buttonShortcut: ""
			sourceImg: "qrc:/qml/img/restoreicon@2x.png"
			onClicked: {
				restore()
			}
			text: qsTr("Restore")
			function restore()
			{
				var state = projectModel.stateListModel.reloadStateFromFromProject(scenarioList.currentIndex)
				restored(state)
				loaded(state)
			}
		}

		ScenarioButton {
			id: saveScenario
			text: qsTr("Save")
			onClicked: {
				projectModel.saveProjectFile()
				saved(state)
			}
			width: 100
			height: 30
			buttonShortcut: ""
			sourceImg: "qrc:/qml/img/saveicon@2x.png"
		}

		ScenarioButton
		{
			id: duplicateScenario
			text: qsTr("Duplicate")
			onClicked: {
				projectModel.stateListModel.duplicateState(scenarioList.currentIndex)
				duplicated(state)
			}
			width: 100
			height: 30
			buttonShortcut: ""
			sourceImg: "qrc:/qml/img/duplicateicon@2x.png"
		}

		ScenarioButton {
			id: addScenario
			width: 100
			height: 30
			buttonShortcut: ""
			sourceImg: "qrc:/qml/img/plus.png"
			onClicked: {
				newStateWin.open()
			}
			text: qsTr("New")
		}
	}

}
