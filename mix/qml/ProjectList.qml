import QtQuick 2.0
import QtQuick.Window 2.0
import QtQuick.Layouts 1.0
import QtQuick.Controls 1.0
import org.ethereum.qml.ProjectModel 1.0

Item {
	property bool renameMode: false;
	ColumnLayout {
		anchors.fill: parent
		Text {
			Layout.fillWidth: true
			color: "blue"
			text: ProjectModel.projectData ? ProjectModel.projectData.title : ""
			horizontalAlignment: Text.AlignHCenter
			visible: !ProjectModel.isEmpty;
		}
		ListView {
			id: projectList
			Layout.fillWidth: true
			Layout.fillHeight: true

			model: ProjectModel.listModel

			delegate: renderDelegate
			highlight: Rectangle {
				color: "lightsteelblue";
			}
			highlightFollowsCurrentItem: true
			focus: true
			clip: true

			onCurrentIndexChanged: {
				if (currentIndex >= 0 && currentIndex < ProjectModel.listModel.count)
					ProjectModel.openDocument(ProjectModel.listModel.get(currentIndex).documentId);
			}
		}
		Menu {
			id: contextMenu
			MenuItem {
				text: qsTr("Rename")
				onTriggered: {
					renameMode = true;
				}
			}
			MenuItem {
				text: qsTr("Delete")
				onTriggered: {
					ProjectModel.removeDocument(projectList.model.get(projectList.currentIndex).documentId);
				}
			}
		}
	}
	Component {
		id: renderDelegate
		Item {
			id: wrapperItem
			height: 20
			width: parent.width
			RowLayout {
				anchors.fill: parent
				visible: !(index === projectList.currentIndex) || !renameMode
				Text {
					id: nameText
					Layout.fillWidth: true
					Layout.fillHeight: true
					text: name
					font.pointSize: 12
					verticalAlignment: Text.AlignBottom
				}
			}

			TextInput {
				id: textInput
				text: nameText.text
				visible: (index === projectList.currentIndex) && renameMode
				MouseArea {
					id: textMouseArea
					anchors.fill: parent
					hoverEnabled: true
					z:2
					onClicked: {
						console.log("clicked");
						textInput.forceActiveFocus();
					}
				}

				onVisibleChanged: {
					if (visible) {
						selectAll();
						forceActiveFocus();
					}
				}

				onAccepted: close(true);
				onCursorVisibleChanged: {
					if (!cursorVisible)
						close(false);
				}
				onFocusChanged: {
					if (!focus)
						close(false);
				}
				function close(accept) {
					renameMode = false;
					if (accept)
						ProjectModel.renameDocument(projectList.model.get(projectList.currentIndex).documentId, textInput.text);
				}
			}
			MouseArea {
				id: mouseArea
				z: 1
				hoverEnabled: false
				anchors.fill: parent
				acceptedButtons: Qt.LeftButton | Qt.RightButton
				onClicked:{
					projectList.currentIndex = index;
					if (mouse.button === Qt.RightButton && !projectList.model.get(index).isContract)
						contextMenu.popup();
				}
			}
			Connections {
				target: ProjectModel
				onProjectLoaded: {
					projectList.currentIndex = 0;
				}
			}
		}
	}
}

