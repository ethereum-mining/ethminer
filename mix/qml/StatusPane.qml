import QtQuick 2.2
import QtQuick.Controls 1.1
import QtQuick.Layouts 1.1
import "js/ErrorLocationFormater.js" as ErrorLocationFormater
import "."

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
			debugImg.state = "active";
		}
		else
		{
			status.state = "error";
			var errorInfo = ErrorLocationFormater.extractErrorInfo(statusPane.result.compilerMessage, true);
			status.text = errorInfo.errorLocation + " " + errorInfo.errorDetail;
			logslink.visible = true;
			debugImg.state = "";
		}
		debugRunActionIcon.enabled = statusPane.result.successful;
	}

	function infoMessage(text)
	{
		status.state = "";
		status.text = text
		logslink.visible = false;
	}


	Connections {
		target:clientModel
		onRunStarted: infoMessage(qsTr("Running transactions..."));
		onRunFailed: infoMessage(qsTr("Error running transactions"));
		onRunComplete: infoMessage(qsTr("Run complete"));
		onNewBlock: infoMessage(qsTr("New block created"));
	}
	Connections {
		target:projectModel
		onDeploymentStarted: infoMessage(qsTr("Running deployment..."));
		onDeploymentError: infoMessage(error);
		onDeploymentComplete: infoMessage(qsTr("Deployment complete"));
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
				font.pointSize: StatusPaneStyle.general.statusFontSize
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
				font.pointSize: StatusPaneStyle.general.logLinkFontSize
				height: 9
				text: qsTr("See Log.")
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
					anchors.rightMargin: 9
					anchors.verticalCenter: parent.verticalCenter
					id: debugImg
					iconSource: "qrc:/qml/img/bugiconinactive.png"
					action: debugRunActionIcon
					states: [
						State{
							name: "active"
							PropertyChanges { target: debugImg; iconSource: "qrc:/qml/img/bugiconactive.png"}
						}
					]
				}
				Action {
					id: debugRunActionIcon
					onTriggered: {
						if (mainContent.rightViewIsVisible())
							mainContent.hideRightView()
						else
							mainContent.startQuickDebugging();
					}
					enabled: false
				}
			}
		}
	}
}
