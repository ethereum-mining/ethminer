import QtQuick 2.2
import QtQuick.Controls 1.1
import QtQuick.Layouts 1.1
import QtQuick.Controls.Styles 1.3
import "js/ErrorLocationFormater.js" as ErrorLocationFormater
import "."

Rectangle {
	id: statusHeader
	objectName: "statusPane"

	function updateStatus(message)
	{
		if (!message)
		{
			status.state = "";
			status.text = qsTr("Compile successfully.");
			logslink.visible = false;
			debugImg.state = "active";
		}
		else
		{
			status.state = "error";
			var errorInfo = ErrorLocationFormater.extractErrorInfo(message, true);
			status.text = errorInfo.errorLocation + " " + errorInfo.errorDetail;
			logslink.visible = true;
			debugImg.state = "";
		}
		debugRunActionIcon.enabled = codeModel.hasContract;
	}

	function infoMessage(text)
	{
		status.state = "";
		status.text = text
		logslink.visible = false;
	}

	function errorMessage(text)
	{
		status.state = "error";
		status.text = text
		logslink.visible = false;
	}

	Connections {
		target:clientModel
		onRunStarted: infoMessage(qsTr("Running transactions..."));
		onRunFailed: errorMessage(qsTr("Error running transactions: " + _message));
		onRunComplete: infoMessage(qsTr("Run complete"));
		onNewBlock: infoMessage(qsTr("New block created"));
	}
	Connections {
		target:projectModel
		onDeploymentStarted: infoMessage(qsTr("Running deployment..."));
		onDeploymentError: errorMessage(error);
		onDeploymentComplete: infoMessage(qsTr("Deployment complete"));
		onDeploymentStepChanged: infoMessage(message);
	}
	Connections {
		target: codeModel
		onCompilationComplete: updateStatus();
		onCompilationError: updateStatus(_error);
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

		Text {
			anchors.verticalCenter: parent.verticalCenter
			anchors.horizontalCenter: parent.horizontalCenter
			font.pointSize: StatusPaneStyle.general.statusFontSize
			height: 15
			font.family: "sans serif"
			objectName: "status"
			wrapMode: Text.WrapAnywhere
			elide: Text.ElideRight
			maximumLineCount: 1
			clip: true
			id: status
			states: [
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
			onTextChanged:
			{
				updateWidth()
				toolTipInfo.tooltip = text;
			}

			function updateWidth()
			{
				if (text.length > 80)
					width = parent.width - 10
				else
					width = undefined
			}
		}

		Button
		{
			anchors.fill: parent
			id: toolTip
			action: toolTipInfo
			text: ""
			style:
				ButtonStyle {
				background:Rectangle {
					color: "transparent"
				}
			}
		}

		Action {
			id: toolTipInfo
			tooltip: ""
		}
	}

	Button
	{
		id: logslink
		anchors.left: statusContainer.right
		anchors.leftMargin: 9
		visible: false
		anchors.verticalCenter: parent.verticalCenter
		action: displayLogAction
		iconSource: "qrc:/qml/img/search_filled.png"
	}

	Action {
		id: displayLogAction
		tooltip: qsTr("Display Log")
		onTriggered: {
			mainContent.displayCompilationErrorIfAny();
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
			Rectangle
			{
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
