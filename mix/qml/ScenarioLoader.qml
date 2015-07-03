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

ColumnLayout
{
	id: blockChainSelector
	signal restored(variant scenario)
	signal saved(variant scenario)
	signal duplicated(variant scenario)
	signal loaded(variant scenario)
	signal renamed(variant scenario)
	spacing: 0
	function init()
	{
		scenarioList.load()
	}

	function needSaveOrReload()
	{
		editStatus.visible = true
	}

	Rectangle
	{
		Layout.fillWidth: true
		Layout.preferredHeight: 30
		color: "transparent"
		Rectangle
		{
			anchors.verticalCenter: parent.verticalCenter
			anchors.horizontalCenter: parent.horizontalCenter
			color: "transparent"
			Label
			{
				anchors.verticalCenter: parent.verticalCenter
				anchors.horizontalCenter: parent.horizontalCenter
				id: scenarioName
				font.bold: true
			}

			TextInput
			{
				id: scenarioNameEdit
				visible: false
				anchors.verticalCenter: parent.verticalCenter
				anchors.horizontalCenter: parent.horizontalCenter
				Keys.onEnterPressed:
				{
					save()
				}

				function edit()
				{
					editIconRect.anchors.left = scenarioNameEdit.right
					editStatus.parent.anchors.left = scenarioNameEdit.right
					scenarioNameEdit.forceActiveFocus()
				}

				function save()
				{
					editIconRect.anchors.left = scenarioName.right
					editStatus.parent.anchors.left = scenarioName.right
					scenarioName.text = scenarioNameEdit.text
					scenarioName.visible = true
					scenarioNameEdit.visible = false
					projectModel.stateListModel.getState(scenarioList.currentIndex).title = scenarioName.text
					projectModel.saveProjectFile()
					saved(state)
					scenarioList.model.get(scenarioList.currentIndex).title = scenarioName.text
					scenarioList.currentIndex = scenarioList.currentIndex
					renamed(projectModel.stateListModel.getState(scenarioList.currentIndex))
				}
			}

			Connections
			{
				target: blockChainSelector
				onLoaded:
				{
					scenarioName.text = scenario.title
					scenarioNameEdit.text = scenario.title
				}
			}

			Rectangle
			{
				anchors.left: scenarioName.right
				anchors.top: scenarioName.top
				anchors.leftMargin: 2
				Layout.preferredWidth: 20
				Text {
					id: editStatus
					text: "*"
					visible: false
				}
			}

			Rectangle
			{
				id: editIconRect
				anchors.top: scenarioName.top
				anchors.topMargin: 6
				anchors.left: scenarioName.right
				anchors.leftMargin: 20
				Image {
					source: "qrc:/qml/img/edittransaction.png"
					width: 30
					fillMode: Image.PreserveAspectFit
					anchors.verticalCenter: parent.verticalCenter
					anchors.horizontalCenter: parent.horizontalCenter
					MouseArea
					{
						anchors.fill: parent
						onClicked:
						{
							scenarioName.visible = !scenarioName.visible
							scenarioNameEdit.visible = !scenarioNameEdit.visible
							if (!scenarioNameEdit.visible)
								scenarioNameEdit.save()
							else
								scenarioNameEdit.edit()
						}
					}
				}
			}
		}
	}

	RowLayout
	{
		Layout.preferredWidth: 560
		anchors.horizontalCenter: parent.horizontalCenter
		Layout.preferredHeight: 50
		spacing: 0

		Row
		{
			Layout.preferredWidth: 100 * 5
			Layout.preferredHeight: 50
			spacing: 15

			Rectangle
			{
				color: "transparent"
				width: 251
				height: 30
				Rectangle
				{
					width: 10
					height: parent.height
					anchors.right: scenarioList.left
					anchors.rightMargin: -8
					radius: 15
				}

				ComboBox
				{
					id: scenarioList
					model: projectModel.stateListModel
					textRole: "title"
					height: parent.height
					width: 150
					onCurrentIndexChanged:
					{
						restoreScenario.restore()
					}

					function load()
					{
						var state = projectModel.stateListModel.getState(currentIndex)
						loaded(state)
					}

					style: ComboBoxStyle {
						background: Rectangle {
							color: "white"
							anchors.fill: parent
						}
						label: Rectangle {
							anchors.fill: parent
							color: "white"
							Label {
								id: comboLabel
								maximumLineCount: 1
								elide: Text.ElideRight
								width: parent.width
								anchors.verticalCenter: parent.verticalCenter
								anchors.horizontalCenter: parent.horizontalCenter
								text: {
									if (projectModel.stateListModel.getState(scenarioList.currentIndex))
										return projectModel.stateListModel.getState(scenarioList.currentIndex).title
									else
										return ""
								}
								Connections {
									target: blockChainSelector
									onLoaded: {
										comboLabel.text = projectModel.stateListModel.getState(scenarioList.currentIndex).title
									}
									onRenamed: {
										comboLabel.text = scenario.title
									}
								}
							}
						}
					}
				}

				Rectangle
				{
					width: 1
					height: parent.height
					anchors.right: addScenario.left
					color: "#ededed"
				}

				ScenarioButton {
					id: addScenario
					anchors.left: scenarioList.right
					width: 100
					height: parent.height
					buttonShortcut: ""
					sourceImg: "qrc:/qml/img/restoreicon@2x.png"
					onClicked: {
						var item = projectModel.stateListModel.createDefaultState();
						item.title = qsTr("New Scenario")
						projectModel.stateListModel.appendState(item)
						projectModel.stateListModel.save()
						scenarioList.currentIndex = projectModel.stateListModel.count - 1
						scenarioNameEdit.edit()
					}
					text: qsTr("New..")
					roundRight: true
					roundLeft: false
				}
			}


			Rectangle
			{
				width: 100 * 3
				height: 30
				color: "transparent"

				Rectangle
				{
					width: 10
					height: parent.height
					anchors.right: restoreScenario.left
					anchors.rightMargin: -4
					radius: 15
				}

				ScenarioButton {
					id: restoreScenario
					width: 100
					height: parent.height
					buttonShortcut: ""
					sourceImg: "qrc:/qml/img/restoreicon@2x.png"
					onClicked: {
						restore()
					}
					text: qsTr("Restore")
					function restore()
					{
						var state = projectModel.stateListModel.reloadStateFromFromProject(scenarioList.currentIndex)
						if (state)
						{
							editStatus.visible = false
							restored(state)
							loaded(state)
						}
					}
					roundRight: false
					roundLeft: true
				}


				Rectangle
				{
					width: 1
					height: parent.height
					anchors.right: saveScenario.left
					color: "#ededed"
				}

				ScenarioButton {
					id: saveScenario
					anchors.left: restoreScenario.right
					text: qsTr("Save")
					onClicked: {
						projectModel.saveProjectFile()
						saved(state)
					}
					width: 100
					height: parent.height
					buttonShortcut: ""
					sourceImg: "qrc:/qml/img/saveicon@2x.png"
					roundRight: false
					roundLeft: false
				}

				Rectangle
				{
					width: 1
					height: parent.height
					anchors.right: duplicateScenario.left
					color: "#ededed"
				}

				ScenarioButton
				{
					id: duplicateScenario
					anchors.left: saveScenario.right
					text: qsTr("Duplicate")
					onClicked: {
						projectModel.stateListModel.duplicateState(scenarioList.currentIndex)
						duplicated(state)
					}
					width: 100
					height: parent.height
					buttonShortcut: ""
					sourceImg: "qrc:/qml/img/duplicateicon@2x.png"
					roundRight: true
					roundLeft: false
				}
			}
		}
	}
}
