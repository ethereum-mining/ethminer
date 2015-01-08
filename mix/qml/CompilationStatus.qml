import QtQuick 2.2
import QtQuick.Controls 1.1
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

	anchors.fill: parent
	gradient: Gradient {
		GradientStop { position: 0.0; color: "#f1f1f1" }
		GradientStop { position: 1.0; color: "#d9d7da" }
	}
	Rectangle {
		anchors.horizontalCenter: parent.horizontalCenter
		anchors.verticalCenter: parent.verticalCenter
		anchors.topMargin: 10
		anchors.bottomMargin: 10
		radius: 3
		width: 500
		height: 30
		color: "#fffcd5"
		Row {
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
						debugModel.debugDeployment();
					}
				}
			}
		}
	}
}
