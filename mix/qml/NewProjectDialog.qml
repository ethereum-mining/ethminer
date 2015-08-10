import QtQuick 2.2
import QtQuick.Controls 1.1
import QtQuick.Layouts 1.1
import QtQuick.Dialogs 1.2
import QtQuick.Window 2.0
import QtQuick.Dialogs 1.1

Item
{
	property alias projectTitle: titleField.text
	readonly property string projectPath: "file://" + pathField.text
	property alias pathFieldText: pathField.text
	signal accepted
	function open() {
		newProjectWin.visible = true;
		titleField.focus = true;
	}

	function close() {
		newProjectWin.visible = false;
	}

	function acceptAndClose() {
		close();
		accepted();
	}

	Dialog {
		id: newProjectWin
		modality: Qt.ApplicationModal
		title: qsTr("New Project");

		width: 640
		height: 120

		visible: false

		contentItem: Rectangle {
			anchors.fill: parent
			GridLayout
			{
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
					Keys.onReturnPressed: {
						if (okButton.enabled)
							acceptAndClose();
					}
				}

				Label {
					text: qsTr("Path")
				}
				RowLayout {
					TextField {
						id: pathField
						Layout.fillWidth: true
						Keys.onReturnPressed: {
							if (okButton.enabled)
								acceptAndClose();
						}
					}
					Button {
						text: qsTr("Browse")
						onClicked:
						{
							newProjectWin.close()
							createProjectFileDialog.open()
						}
					}
				}

				RowLayout
				{
					anchors.bottom: parent.bottom
					anchors.right: parent.right;

					Button {
						id: okButton;
						enabled: titleField.text != "" && pathField.text != ""
						text: qsTr("OK");
						onClicked: {
							acceptAndClose();
						}
					}
					Button {
						text: qsTr("Cancel");
						onClicked: close();
					}
				}
			}
		}
		Component.onCompleted: pathField.text = fileIo.homePath

	}

	QFileDialog {
		id: createProjectFileDialog
		visible: false
		title: qsTr("Please choose a path for the project")
		selectFolder: true
		onAccepted: {
			var u = createProjectFileDialog.fileUrl.toString();
			if (u.indexOf("file://") == 0)
				u = u.substring(7, u.length)
			if (Qt.platform.os == "windows" && u.indexOf("/") == 0)
				u = u.substring(1, u.length);
			pathField.text = u;
			newProjectWin.open()
		}
	}
}
