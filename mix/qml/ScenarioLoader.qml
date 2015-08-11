import QtQuick 2.2
import QtQuick.Controls 1.1
import QtQuick.Controls.Styles 1.1
import QtQuick.Layouts 1.1
import QtQuick.Dialogs 1.2
import QtQuick.Window 2.0
import QtQuick.Dialogs 1.1
import Qt.labs.settings 1.0
import org.ethereum.qml.InverseMouseArea 1.0
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
	signal deleted()
	property alias selectedScenarioIndex: scenarioList.currentIndex
	spacing: 0
	function init()
	{
		scenarioList.load()
	}

	function needSaveOrReload()
	{
	}

	RowLayout
	{
		Layout.preferredWidth: 560
		anchors.horizontalCenter: parent.horizontalCenter
		Layout.preferredHeight: 75
		spacing: 0
		anchors.top: parent.top
		anchors.topMargin: 10

		Row
		{
			Layout.preferredWidth: 100 * 5 + 30
			Layout.preferredHeight: 50
			spacing: 25

			Rectangle
			{
				color: "white"
				width: 251 + 30
				height: 30

				Rectangle
				{
					id: left
					width: 10
					height: parent.height
					anchors.left: parent.left
					anchors.leftMargin: -5
					radius: 15
				}

				Image {
					source: "qrc:/qml/img/edittransaction.png"
					height: parent.height - 10
					fillMode: Image.PreserveAspectFit
					anchors.left: parent.left
					anchors.top: parent.top
					anchors.topMargin: 4
					id: editImg
					MouseArea
					{
						anchors.fill: parent
						onClicked:
						{
							scenarioNameEdit.toggleEdit()
						}
					}
				}

				Image {
					source: "qrc:/qml/img/delete_sign.png"
					height: parent.height - 16
					fillMode: Image.PreserveAspectFit
					id: deleteImg
					anchors.left: editImg.right
					anchors.top: parent.top
					anchors.topMargin: 8
					visible: projectModel.stateListModel.count > 1
					MouseArea
					{
						anchors.fill: parent
						onClicked:
						{
							if (projectModel.stateListModel.count > 1)
							{
								projectModel.stateListModel.deleteState(scenarioList.currentIndex)
								scenarioList.init()
							}
						}
					}
				}

				Label
				{

					MouseArea
					{
						anchors.fill: parent
						onClicked:
						{
							if (projectModel.stateListModel.count > 1)
							{
								projectModel.stateListModel.deleteState(scenarioList.currentIndex)
								scenarioList.init()
							}
						}
					}
				}

				Connections
				{
					target: projectModel.stateListModel
					onStateDeleted: {
						scenarioList.init()
					}
				}

				ComboBox
				{
					id: scenarioList
					anchors.left: deleteImg.right
					anchors.leftMargin: 2
					model: projectModel.stateListModel
					anchors.top: parent.top
					textRole: "title"
					height: parent.height
					width: 150
					signal updateView()

					onCurrentIndexChanged:
					{
						restoreScenario.restore()
					}

					function init()
					{
						scenarioList.currentIndex = 0
						deleted()
					}

					function load()
					{
						var state = projectModel.stateListModel.getState(currentIndex)
						if (state)
							loaded(state)
					}

					style: ComboBoxStyle {
						id: style
						background: Rectangle {
							color: "white"
							anchors.fill: parent
						}
						label: Rectangle {
							property alias label: comboLabel
							anchors.fill: parent
							color: "white"
							Label {
								id: comboLabel
								maximumLineCount: 1
								elide: Text.ElideRight
								width: parent.width
								anchors.verticalCenter: parent.verticalCenter
								anchors.left: parent.left
								Component.onCompleted:
								{
									comboLabel.updateLabel()
								}

								function updateLabel()
								{
									comboLabel.text = ""
									if (scenarioList.currentIndex > - 1 && scenarioList.currentIndex < projectModel.stateListModel.count)
										comboLabel.text = projectModel.stateListModel.getState(scenarioList.currentIndex).title
								}

								Connections {
									target: blockChainSelector
									onLoaded: {
										if (projectModel.stateListModel.count > 0)
											comboLabel.text = projectModel.stateListModel.getState(scenarioList.currentIndex).title
										else
											return ""
									}
									onRenamed: {
										comboLabel.text = scenario.title
										scenarioNameEdit.text = scenario.title
									}
									onDeleted: {
										comboLabel.updateLabel()
									}
								}
							}
						}
					}
				}

				TextField
				{
					id: scenarioNameEdit
					anchors.left: deleteImg.right
					anchors.leftMargin: 2
					height: parent.height
					z: 5
					visible: false
					width: 150
					Keys.onEnterPressed:
					{
						toggleEdit()
					}

					Keys.onReturnPressed:
					{
						toggleEdit()
					}

					function toggleEdit()
					{
						scenarioList.visible = !scenarioList.visible
						scenarioNameEdit.visible = !scenarioNameEdit.visible
						if (!scenarioNameEdit.visible)
							scenarioNameEdit.save()
						else
						{
							scenarioNameEdit.text = projectModel.stateListModel.getState(scenarioList.currentIndex).title
							scenarioNameEdit.forceActiveFocus()
							outsideClick.active = true
						}
					}

					function save()
					{
						outsideClick.active = false
						projectModel.stateListModel.getState(scenarioList.currentIndex).title = scenarioNameEdit.text
						projectModel.saveProjectFile()
						saved(state)
						scenarioList.model.get(scenarioList.currentIndex).title = scenarioNameEdit.text
						scenarioList.currentIndex = scenarioList.currentIndex
						renamed(projectModel.stateListModel.getState(scenarioList.currentIndex))
					}

					style: TextFieldStyle {
						background: Rectangle {
									radius: 2
									implicitWidth: 100
									implicitHeight: 30
									color: "white"
									border.color: "#cccccc"
									border.width: 1
								}
					}

					InverseMouseArea {
						id: outsideClick
						anchors.fill: parent
						active: false
						onClickedOutside: {
							scenarioNameEdit.toggleEdit()
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
						var state = projectModel.stateListModel.reloadStateFromProject(scenarioList.currentIndex)
						if (state)
						{
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
