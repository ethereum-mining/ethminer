import QtQuick 2.2
import QtQuick.Controls 1.1
import QtQuick.Layouts 1.1
import "js/ErrorLocationFormater.js" as ErrorLocationFormater

Rectangle {
	id: statusHeader
	objectName: "statusPane"

	function updateStatus()
	{
		if (statusPane.result.successful)
		{
			status.state = "";
			status.text = qsTr("Compile without errors.");
			logslink.visible = false;
		}
		else
		{
			status.state = "error";
			var errorInfo = ErrorLocationFormater.extractErrorInfo(statusPane.result.compilerMessage, true);
			status.text = errorInfo.errorLocation + " " + errorInfo.errorDetail;
			logslink.visible = true;
		}
		debugRunActionIcon.enabled = statusPane.result.successful;
	}

	color: "transparent"
	anchors.fill: parent
	Rectangle {
		id: statusContainer
		anchors.horizontalCenter: parent.horizontalCenter
		anchors.verticalCenter: parent.verticalCenter
		radius: 3
		width: 500
		height: 30
		color: "#fcfbfc"
		RowLayout {
			anchors.horizontalCenter: parent.horizontalCenter
			anchors.verticalCenter: parent.verticalCenter
			spacing: 5

			Text {
				font.pointSize: 10
				height: 9
				font.family: "sans serif"
				objectName: "status"
				id: status
				states:[
					State {
						name: "error"
						PropertyChanges {
							target: status
							color: "red"
						}
						PropertyChanges {
							target: statusContainer
							color: "#fffcd5"
						}
					}
				]
			}

			Text {
				visible: false
				font.pointSize: 9
				height: 9
				text: qsTr("See log.")
				font.family: "Monospace"
				objectName: "status"
				id: logslink
				color: "#8c8a74"
				MouseArea {
					anchors.fill: parent
					onClicked: {
						mainContent.ensureRightView();
					}
				}
			}
		}
	}

	Rectangle
	{
		color: "transparent"
		width: 100
		height: parent.height
		anchors.top: statusHeader.top
		anchors.right: statusHeader.right
		RowLayout
		{
			anchors.fill: parent
			Rectangle {
				color: "transparent"
				anchors.fill: parent
				Button
				{
					anchors.right: parent.right
					anchors.rightMargin: 7
					anchors.verticalCenter: parent.verticalCenter
					id: debugImg
					iconSource: "qrc:/qml/img/bugiconinactive.png"
					action: debugRunActionIcon
				}
				Action {
					id: debugRunActionIcon
					onTriggered: {
						mainContent.ensureRightView();
						clientModel.debugDeployment();
					}
					enabled: false
				}
			}
		}
	}
}
