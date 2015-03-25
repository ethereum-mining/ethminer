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
			errorMessage(status.text, "Compilation");
		}
		debugRunActionIcon.enabled = codeModel.hasContract;
	}

	function infoMessage(text, type)
	{
		status.state = "";
		status.text = text
		logPane.push("info", type, text);
	}

	function warningMessage(text, type)
	{
		status.state = "warning";
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
		onJavaScriptMessage:
		{
			if (_level === 0)
				infoMessage(_content, "JavaScript")
			else
			{
				var message = _sourceId.substring(_sourceId.lastIndexOf("/") + 1) + " - " + qsTr("line") + " " + _lineNb + " - " + _content;
				if (_level === 1)
					warningMessage(message, "JavaScript")
				else
					errorMessage(message, "JavaScript")
			}
		}
	}

	Connections {
		target:clientModel
		onRunStarted: infoMessage(qsTr("Running transactions..."), "Run");
		onRunFailed: errorMessage(format(_message), "Run");
		onRunComplete: infoMessage(qsTr("Run complete"), "Run");
		onNewBlock: infoMessage(qsTr("New block created"), "State");

		function format(_message)
		{
			var formatted = _message.match(/(?:<dev::eth::)(.+)(?:>)/);
			if (formatted === null)
				formatted = _message.match(/(?:<dev::)(.+)(?:>)/);
			if (formatted.length > 1)
				formatted = formatted[1];
			else
				return _message;
			var exceptionInfos = _message.match(/(?:tag_)(.+)/g);
			if (exceptionInfos !== null && exceptionInfos.length > 0)
				formatted += ": "
			for (var k in exceptionInfos)
				formatted += " " + exceptionInfos[k].replace("*]", "").replace("tag_", "").replace("=", "");
			return formatted;
		}
	}
	Connections {
		target:projectModel
		onDeploymentStarted: infoMessage(qsTr("Running deployment..."), "Deployment");
		onDeploymentError: errorMessage(error, "Deployment");
		onDeploymentComplete: infoMessage(qsTr("Deployment complete"), "Deployment");
		onDeploymentStepChanged: infoMessage(message, "Deployment");
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
		states: [
			State {
				name: "logsOpened"
				PropertyChanges {
					target: statusContainer
					border.color: "#5391d8"
					border.width: 1
				}
			},
			State {
				name: "logsClosed"
				PropertyChanges {
					target: statusContainer
					border.color: "#5391d8"
					border.width: 0
				}
			}
		]

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
				},
				State {
					name: "warning"
					PropertyChanges {
						target: status
						color: "orange"
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
			MouseArea {
				anchors.fill: parent
				onClicked: {
					logsContainer.toggle();
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
				{
					statusContainer.state = "logsClosed";
					logsContainer.state = "closed"
				}
				else
				{
					statusContainer.state = "logsOpened";
					logsContainer.state = "opened";
					logsContainer.focus = true;
					forceActiveFocus();
				}
			}

			id: logsContainer
			width: 1000
			height: 0
			anchors.topMargin: 10
			anchors.top: statusContainer.bottom
			anchors.horizontalCenter: parent.horizontalCenter
			visible: false
			radius: 5
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

	Rectangle
	{
		color: "transparent"
		width: 100
		height: parent.height
		anchors.top: parent.top
		anchors.right: parent.right
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
