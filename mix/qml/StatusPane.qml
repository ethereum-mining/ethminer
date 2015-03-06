import QtQuick 2.2
import QtQuick.Controls 1.1
import QtQuick.Layouts 1.1
import QtQuick.Controls.Styles 1.3
import "js/ErrorLocationFormater.js" as ErrorLocationFormater
import "."

Rectangle {
	id: statusHeader
	objectName: "statusPane"
	property variant webPreview

	function updateStatus(message)
	{
		if (!message)
		{
			status.state = "";
			status.text = qsTr("Compile successfully.");
			debugImg.state = "active";
		}
		else
		{
			status.state = "error";
			var errorInfo = ErrorLocationFormater.extractErrorInfo(message, true);
			status.text = errorInfo.errorLocation + " " + errorInfo.errorDetail;
			debugImg.state = "";
		}
		debugRunActionIcon.enabled = codeModel.hasContract;
	}

	function infoMessage(text, type)
	{
		status.state = "";
		status.text = text
		logPane.push("info",type, text);
	}

	function warningMessage(text, type)
	{
		status.state = "";
		status.text = text
		logPane.push("warning", type, text);
	}

	function errorMessage(text, type)
	{
		status.state = "error";
		status.text = text
		logPane.push("error", type, text);
	}

	Connections {
		target: webPreview
		onJavaScriptErrorMessage: errorMessage(_content, "javascript")
		onJavaScriptWarningMessage: warningMessage(_content, "javascript")
		onJavaScriptInfoMessage: infoMessage(_content, "javascript")
	}

	Connections {
		target:clientModel
		onRunStarted: infoMessage(qsTr("Running transactions..."), "run");
		onRunFailed: errorMessage(qsTr("Error running transactions: " + _message), "run");
		onRunComplete: infoMessage(qsTr("Run complete"), "run");
		onNewBlock: infoMessage(qsTr("New block created"), "state");
	}
	Connections {
		target:projectModel
		onDeploymentStarted: infoMessage(qsTr("Running deployment..."), "deployment");
		onDeploymentError: errorMessage(error, "deployment");
		onDeploymentComplete: infoMessage(qsTr("Deployment complete"), "deployment");
		onDeploymentStepChanged: infoMessage(message, "deployment");
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

		Rectangle
		{
			function toggle()
			{
				if (logsContainer.state === "opened")
					logsContainer.state = "closed"
				else
					logsContainer.state = "opened";
			}

			id: logsContainer
			width: 1000
			height: 0
			anchors.topMargin: 2
			anchors.top: statusContainer.bottom
			anchors.horizontalCenter: parent.horizontalCenter
			visible: false
			Component.onCompleted:
			{
				var top = logsContainer;
				while (top.parent)
					top = top.parent
				var coordinates = logsContainer.mapToItem(top, 0, 0)
				logsContainer.parent = top;
				logsContainer.x = coordinates.x
				logsContainer.y = coordinates.y
			}
			LogsPane
			{
				id: logPane
			}
			states: [
				State {
					name: "opened";
					PropertyChanges { target: logsContainer; height: 500; visible: true }
				},
				State {
					name: "closed";
					PropertyChanges { target: logsContainer; height: 0; visible: false }
				}
			]
			transitions: Transition {
					 NumberAnimation { properties: "height"; easing.type: Easing.InOutQuad; duration: 200 }
					 NumberAnimation { properties: "visible"; easing.type: Easing.InOutQuad; duration: 200 }
				 }
		}
	}

	Button
	{
		id: logslink
		anchors.left: statusContainer.right
		anchors.leftMargin: 9
		anchors.verticalCenter: parent.verticalCenter
		action: displayLogAction
		iconSource: "qrc:/qml/img/search_filled.png"
	}

	Action {
		id: displayLogAction
		tooltip: qsTr("Display Log")
		onTriggered: {
			logsContainer.toggle();
			//if (status.state === "error" && logPane.front().type === "run")
			//	mainContent.displayCompilationErrorIfAny();
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
