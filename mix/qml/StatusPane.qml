import QtQuick 2.2
import QtQuick.Controls 1.1
import QtQuick.Layouts 1.1
import QtQuick.Controls.Styles 1.3
import org.ethereum.qml.InverseMouseArea 1.0
import "js/ErrorLocationFormater.js" as ErrorLocationFormater
import "."

Rectangle {
	id: statusHeader
	objectName: "statusPane"
	property variant webPreview
	property alias currentStatus: logPane.currentStatus
	function updateStatus(message)
	{
		if (!message)
		{
			status.state = "";
			status.text = qsTr("Compile successfully.");
			debugImg.state = "active";
			currentStatus = { "type": "Comp", "date": Qt.formatDateTime(new Date(), "hh:mm:ss"), "content": status.text, "level": "info" }
		}
		else
		{
			status.state = "error";
			var errorInfo = ErrorLocationFormater.extractErrorInfo(message, true);
			status.text = errorInfo.errorLocation + " " + errorInfo.errorDetail;
			debugImg.state = "";
			currentStatus = { "type": "Comp", "date": Qt.formatDateTime(new Date(), "hh:mm:ss"), "content": status.text, "level": "error" }
		}
	}

	function infoMessage(text, type)
	{
		status.state = "";
		status.text = text
		logPane.push("info", type, text);
		currentStatus = { "type": type, "date": Qt.formatDateTime(new Date(), "hh:mm:ss"), "content": text, "level": "info" }
	}

	function warningMessage(text, type)
	{
		status.state = "warning";
		status.text = text
		logPane.push("warning", type, text);
		currentStatus = { "type": type, "date": Qt.formatDateTime(new Date(), "hh:mm:ss"), "content": text, "level": "warning" }
	}

	function errorMessage(text, type)
	{
		status.state = "error";
		status.text = text;
		logPane.push("error", type, text);
		currentStatus = { "type": type, "date": Qt.formatDateTime(new Date(), "hh:mm:ss"), "content": text, "level": "error" }
	}

	function clear()
	{
		status.state = "";
		status.text = "";
	}

	StatusPaneStyle {
		id: statusPaneStyle
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
		onRunStarted:
		{
			logPane.clear()
			infoMessage(qsTr("Running transactions..."), "Run");
		}
		onRunFailed: errorMessage(format(_message), "Run");
		onRunComplete: infoMessage(qsTr("Run complete"), "Run");
		onNewBlock: infoMessage(qsTr("New block created"), "State");

		function format(_message)
		{
			var formatted = _message.match(/(?:<dev::eth::)(.+)(?:>)/);
			if (!formatted)
				formatted = _message.match(/(?:<dev::)(.+)(?:>)/);
			if (formatted && formatted.length > 1)
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
		target: codeModel
		onCompilationComplete:
		{
			goToLine.visible = false;
			updateStatus();
		}

		onCompilationError:
		{
			goToLine.visible = true
			updateStatus(_error);
		}
	}

	color: "transparent"
	anchors.fill: parent

	Rectangle {
		id: statusContainer
		anchors.horizontalCenter: parent.horizontalCenter
		anchors.verticalCenter: parent.verticalCenter
		radius: 3
		width: 600
		height: 30
		color: "#fcfbfc"
		Text {
			anchors.verticalCenter: parent.verticalCenter
			anchors.horizontalCenter: parent.horizontalCenter
			font.pointSize: appStyle.absoluteSize(-1)
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
				if (text.length > 100)
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
			z: 3;
			style:
				ButtonStyle {
				background:Rectangle {
					color: "transparent"
				}
			}
			MouseArea {
				anchors.fill: parent
				onClicked: {
					var globalCoord = goToLineBtn.mapToItem(statusContainer, 0, 0);
					if (mouseX > globalCoord.x
							&& mouseX < globalCoord.x + goToLineBtn.width
							&& mouseY > globalCoord.y
							&& mouseY < globalCoord.y + goToLineBtn.height)
						goToCompilationError.trigger(goToLineBtn);
					else
						logsContainer.toggle();
				}
			}
		}

		Rectangle
		{
			visible: false
			color: "transparent"
			width: 40
			height: parent.height
			anchors.top: parent.top
			anchors.left: status.right
			anchors.leftMargin: 15
			id: goToLine
			RowLayout
			{
				anchors.fill: parent
				Rectangle
				{
					color: "transparent"
					anchors.fill: parent
					Button
					{
						z: 4
						anchors.centerIn: parent
						id: goToLineBtn
						text: ""
						width: 30
						height: 30
						action: goToCompilationError
						style: ButtonStyle {
							background: Rectangle {
								color: "transparent"

								Image {
									source: "qrc:/qml/img/warningicon.png"
									height: 30
									width: 30
									sourceSize.width: 30
									sourceSize.height: 30
									anchors.centerIn: parent
								}
							}
						}
					}
				}
			}
		}

		Action {
			id: toolTipInfo
			tooltip: ""
		}

		Rectangle
		{
			id: logsShadow
			width: logsContainer.width + 5
			height: 0
			opacity: 0.3
			clip: true
			anchors.top: logsContainer.top
			anchors.margins: 4
			Rectangle {
				color: "gray"
				anchors.top: parent.top
				radius: 10
				id: roundRect
				height: 400
				width: parent.width
			}
		}



		Rectangle
		{
			InverseMouseArea
			{
				id: outsideClick
				anchors.fill: parent
				active: false
				onClickedOutside: {
					logsContainer.toggle();
				}
			}

			function toggle()
			{
				if (logsContainer.state === "opened")
				{
					statusContainer.visible = true
					logsContainer.state = "closed"
				}
				else
				{
					statusContainer.visible = false
					logsContainer.state = "opened";
					logsContainer.focus = true;
					forceActiveFocus();
					calCoord()
					move()
				}
			}

			id: logsContainer
			width: 750
			anchors.top: statusContainer.bottom
			anchors.topMargin: 4
			visible: false
			radius: 10

			function calCoord()
			{
				if (!logsContainer.parent.parent)
					return
				var top = logsContainer;
				while (top.parent)
					top = top.parent
				var coordinates = logsContainer.mapToItem(top, 0, 0);
				logsContainer.parent = top;
				logsShadow.parent = top;
				top.onWidthChanged.connect(move)
				top.onHeightChanged.connect(move)
			}

			function move()
			{
				var statusGlobalCoord = status.mapToItem(null, 0, 0);
				logsContainer.x = statusGlobalCoord.x - logPane.contentXPos
				logsShadow.x = statusGlobalCoord.x - logPane.contentXPos
				logsShadow.z = 1
				logsContainer.z = 2
				if (Qt.platform.os === "osx")
				{
					logsContainer.y = statusGlobalCoord.y;
					logsShadow.y = statusGlobalCoord.y;
				}
			}

			LogsPaneStyle {
				id: logStyle
			}

			LogsPane
			{
				id: logPane;
				statusPane: statusHeader
				onContentXPosChanged:
				{
					parent.move();
				}
			}

			states: [
				State {
					name: "opened";
					PropertyChanges { target: logsContainer; height: 500; visible: true }
					PropertyChanges { target: logsShadow; height: 500; visible: true }
					PropertyChanges { target: outsideClick; active: true }

				},
				State {
					name: "closed";
					PropertyChanges { target: logsContainer; height: 0; visible: false }
					PropertyChanges { target: statusContainer; width: 600; height: 30 }
					PropertyChanges { target: outsideClick; active: false }
					PropertyChanges { target: logsShadow; height: 0; visible: false }
				}
			]
			transitions: Transition {
					 NumberAnimation { properties: "height"; easing.type: Easing.InOutQuad; duration: 200 }
					 NumberAnimation { target: logsContainer;  properties: "visible"; easing.type: Easing.InOutQuad; duration: 200 }
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
					text: ""
					iconSource: "qrc:/qml/img/bugiconactive.png"
					action: showHideRightPanelAction
				}
			}
		}
	}
}
