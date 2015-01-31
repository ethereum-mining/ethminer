import QtQuick 2.0
import QtQuick.Window 2.0
import QtQuick.Layouts 1.0
import QtQuick.Controls 1.0

Item {
	property bool renameMode: false;
	ColumnLayout {
		anchors.fill: parent
		Text {
			Layout.fillWidth: true
			color: "blue"
			text: projectModel.projectTitle
			horizontalAlignment: Text.AlignHCenter
			visible: !projectModel.isEmpty;
		}
		ListView {
			id: projectList
			Layout.fillWidth: true
			Layout.fillHeight: true

			model: projectModel.listModel

			delegate: renderDelegate
			highlight: Rectangle {
				color: "lightsteelblue";
			}
			highlightFollowsCurrentItem: true
			focus: true
			clip: true

			onCurrentIndexChanged: {
				if (currentIndex >= 0 && currentIndex < projectModel.listModel.count)
					projectModel.openDocument(projectModel.listModel.get(currentIndex).documentId);
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
					projectModel.removeDocument(projectList.model.get(projectList.currentIndex).documentId);
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
						projectModel.renameDocument(projectList.model.get(projectList.currentIndex).documentId, textInput.text);
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
		}
	}
	Connections {
		target: projectModel
		onProjectLoaded: {
			projectList.currentIndex = 0;
			if (projectList.currentIndex >= 0 && projectList.currentIndex < projectModel.listModel.count)
				projectModel.openDocument(projectModel.listModel.get(projectList.currentIndex).documentId);

		}
		onProjectClosed: {
			projectList.currentIndex = -1;
		}
		onDocumentOpened: {
			if (projectList.currentItem.documentId !== document.documentId)
				projectList.currentIndex = projectModel.getDocumentIndex(document.documentId);

		}
	}
}

