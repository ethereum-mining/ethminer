import QtQuick 2.2
import QtQuick.Controls 1.1
import QtQuick.Layouts 1.1
import "js/ErrorLocationFormater.js" as ErrorLocationFormater

Rectangle {
	id: constantCompilationStatus
	objectName: "constantCompilationStatus"
	function update()
	{
		if (constantCompilation.result.successfull)
		{
			image.state = "";
			status.state = "";
			status.text = qsTr("Compile without errors.");
			logslink.visible = false;
		}
		else
		{
			image.state = "error";
			status.state = "error";
			var errorInfo = ErrorLocationFormater.extractErrorInfo(constantCompilation.result.compilerMessage, true);
			status.text = errorInfo.errorLocation + " " + errorInfo.errorDetail;
			logslink.visible = true;
		}
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
			Image
			{
				source: "qrc:/qml/img/compilsuceeded.png"
				id: image
				states:[
					State {
						name: "error"
						PropertyChanges {
							target: image
							source: "qrc:/qml/img/compilfailed.png"
						}
						PropertyChanges {
							target: statusContainer
							color: "#fffcd5"
						}
					}
				]
			}

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
						debugModel.updateDebugPanel();
					}
				}
			}
		}
	}
}
