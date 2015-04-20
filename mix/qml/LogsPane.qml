import QtQuick 2.0
import QtQuick.Layouts 1.0
import QtQuick.Controls 1.1
import QtQuick.Controls.Styles 1.3
import org.ethereum.qml.SortFilterProxyModel 1.0

Rectangle
{
	property variant statusPane
	property variant currentStatus
	property int contentXPos: logStyle.generic.layout.dateWidth + logStyle.generic.layout.typeWidth - 70

	function clear()
	{
		logsModel.clear();
		statusPane.clear();
		currentStatus = undefined;
	}

	function push(_level, _type, _content)
	{
		_content = _content.replace(/\n/g, " ")
		logsModel.insert(0, { "type": _type, "date": Qt.formatDateTime(new Date(), "hh:mm:ss"), "content": _content, "level": _level });
	}

	onVisibleChanged:
	{
		if (currentStatus && visible && (logsModel.count === 0 || logsModel.get(0).content !== currentStatus.content || logsModel.get(0).date !== currentStatus.date))
			logsModel.insert(0, { "type": currentStatus.type, "date": currentStatus.date, "content": currentStatus.content, "level": currentStatus.level });
		else if (!visible)
		{
			for (var k = 0; k < logsModel.count; k++)
			{
				if (logsModel.get(k).type === "Comp") //do not keep compilation logs.
					logsModel.remove(k);
			}
		}
	}

	anchors.fill: parent
	radius: 10
	color: "transparent"
	id: logsPane
	Column {
		z: 2
		height: parent.height - rowAction.height
		width: parent.width
		spacing: 0
		ListModel {
			id: logsModel
		}

		ScrollView
		{
			id: scrollView
			height: parent.height
			width: parent.width
			horizontalScrollBarPolicy: Qt.ScrollBarAlwaysOff
			Column
			{
				id: logsRect
				spacing: 0
				Repeater {
					id: logsRepeater
					clip: true
					property string frontColor: "transparent"
					model: SortFilterProxyModel {
						id: proxyModel
						source: logsModel
						property var roles: ["-", "javascript", "run", "state", "comp", "deployment"]

						Component.onCompleted: {
							filterType = regEx(proxyModel.roles);
						}

						function search(_value)
						{
							filterContent = _value;
						}

						function toogleFilter(_value)
						{
							var count = roles.length;
							for (var i in roles)
							{
								if (roles[i] === _value)
								{
									roles.splice(i, 1);
									break;
								}
							}
							if (count === roles.length)
								roles.push(_value);

							filterType = regEx(proxyModel.roles);
						}

						function regEx(_value)
						{
							return "(?:" + roles.join('|') + ")";
						}

						filterType: "(?:javascript|run|state|comp|deployment)"
						filterContent: ""
						filterSyntax: SortFilterProxyModel.RegExp
						filterCaseSensitivity: Qt.CaseInsensitive
					}

					Rectangle
					{
						width: logStyle.generic.layout.dateWidth + logStyle.generic.layout.contentWidth + logStyle.generic.layout.typeWidth
						height: 30
						color:
						{
							var cl;
							if (level === "warning" || level === "error")
								cl = logStyle.generic.layout.errorColor;
							else
								cl = index % 2 === 0 ? "transparent" : logStyle.generic.layout.logAlternateColor;
							if (index === 0)
								logsRepeater.frontColor = cl;
							return cl;
						}

						Component.onCompleted:
						{
							logsPane.contentXPos = logContent.x
						}

						MouseArea
						{
							anchors.fill: parent
							onClicked:
							{
								if (logContent.elide === Text.ElideNone)
								{
									logContent.elide = Text.ElideRight;
									logContent.wrapMode = Text.NoWrap
									parent.height = 30;
								}
								else
								{
									logContent.elide = Text.ElideNone;
									logContent.wrapMode = Text.WordWrap;
									parent.height = logContent.lineCount * 30;
								}
							}
						}


						DefaultLabel {
							id: dateLabel
							text: date;
							font.family: logStyle.generic.layout.logLabelFont
							width: logStyle.generic.layout.dateWidth
							font.pointSize: appStyle.absoluteSize(-1)
							clip: true
							elide: Text.ElideRight
							anchors.left: parent.left
							anchors.leftMargin: 15
							anchors.verticalCenter: parent.verticalCenter
							color: {
								parent.getColor(level);
							}
						}

						DefaultLabel {
							text: type;
							id: typeLabel
							font.family: logStyle.generic.layout.logLabelFont
							width: logStyle.generic.layout.typeWidth
							clip: true
							elide: Text.ElideRight
							font.pointSize: appStyle.absoluteSize(-1)
							anchors.left: dateLabel.right
							anchors.leftMargin: 2
							anchors.verticalCenter: parent.verticalCenter
							color: {
								parent.getColor(level);
							}
						}

						Text {
							id: logContent
							text: content;
							font.family: logStyle.generic.layout.logLabelFont
							width: logStyle.generic.layout.contentWidth
							font.pointSize: appStyle.absoluteSize(-1)
							anchors.verticalCenter: parent.verticalCenter
							elide: Text.ElideRight
							anchors.left: typeLabel.right
							anchors.leftMargin: 2
							color: {
								parent.getColor(level);
							}
						}

						function getColor()
						{
							if (level === "error")
								return "red";
							else if (level === "warning")
								return "orange";
							else
								return "#808080";
						}
					}
				}
			}
		}

		Component {
			id: itemDelegate
			DefaultLabel {
				text: styleData.value;
				font.family: logStyle.generic.layout.logLabelFont
				font.pointSize: appStyle.absoluteSize(-1)
				color: {
					if (proxyModel.get(styleData.row).level === "error")
						return "red";
					else if (proxyModel.get(styleData.row).level === "warning")
						return "orange";
					else
						return "#808080";
				}
			}
		}
	}

	Rectangle
	{
		gradient: Gradient {
			GradientStop { position: 0.0; color: "#f1f1f1" }
			GradientStop { position: 1.0; color: "#d9d7da" }
		}
		Layout.preferredHeight: logStyle.generic.layout.headerHeight
		height: logStyle.generic.layout.headerHeight
		width: logsPane.width
		anchors.bottom: parent.bottom
		Row
		{
			id: rowAction
			anchors.leftMargin: logStyle.generic.layout.leftMargin
			anchors.left: parent.left
			spacing: logStyle.generic.layout.headerButtonSpacing
			height: parent.height
			Rectangle
			{
				color: "transparent"
				height: parent.height
				width: 40
				DefaultLabel
				{
					width: 40
					elide: Text.ElideRight
					anchors.verticalCenter: parent.verticalCenter
					color: logStyle.generic.layout.logLabelColor
					font.pointSize: appStyle.absoluteSize(-3)
					font.family: logStyle.generic.layout.logLabelFont
					text: qsTr("Show:")
				}
			}

			Rectangle {
				anchors.verticalCenter: parent.verticalCenter
				width: 1;
				height: parent.height
				color: logStyle.generic.layout.buttonSeparatorColor1
			}

			Rectangle {
				anchors.verticalCenter: parent.verticalCenter
				width: 2;
				height: parent.height
				color: logStyle.generic.layout.buttonSeparatorColor2
			}

			ToolButton {
				id: javascriptButton
				checkable: true
				height: logStyle.generic.layout.headerButtonHeight
				width: 30
				anchors.verticalCenter: parent.verticalCenter
				checked: true
				onCheckedChanged: {
					proxyModel.toogleFilter("javascript")
				}
				tooltip: qsTr("JavaScript")
				style:
					ButtonStyle {
					label:
						Item {
						Rectangle
						{
							width: labelJs.width
							height: labelJs.height
							anchors.centerIn: parent
							color: "transparent"
							DefaultLabel {
								id: labelJs
								width: 15
								elide: Text.ElideRight
								anchors.horizontalCenter: parent.horizontalCenter
								font.family: logStyle.generic.layout.logLabelFont
								font.pointSize: appStyle.absoluteSize(-3)
								color: logStyle.generic.layout.logLabelColor
								text: qsTr("JS")
							}
						}
					}
					background:
						Rectangle {
						color: javascriptButton.checked ? logStyle.generic.layout.buttonSelected : "transparent"
					}
				}
			}

			Rectangle {
				anchors.verticalCenter: parent.verticalCenter
				width: 1;
				height: parent.height
				color: logStyle.generic.layout.buttonSeparatorColor1
			}

			Rectangle {
				anchors.verticalCenter: parent.verticalCenter
				width: 2;
				height: parent.height
				color: logStyle.generic.layout.buttonSeparatorColor2
			}

			ToolButton {
				id: runButton
				checkable: true
				height: logStyle.generic.layout.headerButtonHeight
				width: 40
				anchors.verticalCenter: parent.verticalCenter
				checked: true
				onCheckedChanged: {
					proxyModel.toogleFilter("run")
				}
				tooltip: qsTr("Run")
				style:
					ButtonStyle {
					label:
						Item {
						Rectangle
						{
							width: labelRun.width
							height: labelRun.height
							anchors.centerIn: parent
							color: "transparent"
							DefaultLabel {
								id: labelRun
								width: 25
								anchors.horizontalCenter: parent.horizontalCenter
								elide: Text.ElideRight
								font.family: logStyle.generic.layout.logLabelFont
								font.pointSize: appStyle.absoluteSize(-3)
								color: logStyle.generic.layout.logLabelColor
								text: qsTr("Run")
							}
						}
					}
					background:
						Rectangle {
						color: runButton.checked ? logStyle.generic.layout.buttonSelected : "transparent"
					}
				}
			}

			Rectangle {
				anchors.verticalCenter: parent.verticalCenter
				width: 1;
				height: parent.height
				color: logStyle.generic.layout.buttonSeparatorColor1
			}

			Rectangle {
				anchors.verticalCenter: parent.verticalCenter
				width: 2;
				height: parent.height
				color: logStyle.generic.layout.buttonSeparatorColor2
			}

			ToolButton {
				id: stateButton
				checkable: true
				height: logStyle.generic.layout.headerButtonHeight
				anchors.verticalCenter: parent.verticalCenter
				width: 40
				checked: true
				onCheckedChanged: {
					proxyModel.toogleFilter("state")
				}
				tooltip: qsTr("State")
				style:
					ButtonStyle {
					label:
						Item {
						Rectangle
						{
							width: labelState.width
							height: labelState.height
							anchors.centerIn: parent
							color: "transparent"
							DefaultLabel {
								id: labelState
								width: 30
								anchors.horizontalCenter: parent.horizontalCenter
								elide: Text.ElideRight
								font.family: logStyle.generic.layout.logLabelFont
								font.pointSize: appStyle.absoluteSize(-3)
								color: logStyle.generic.layout.logLabelColor
								text: qsTr("State")
							}
						}
					}
					background:
						Rectangle {
						color: stateButton.checked ? logStyle.generic.layout.buttonSelected : "transparent"
					}
				}
			}

			Rectangle {
				anchors.verticalCenter: parent.verticalCenter
				width: 1;
				height: parent.height
				color: logStyle.generic.layout.buttonSeparatorColor1
			}

			Rectangle {
				anchors.verticalCenter: parent.verticalCenter
				width: 2;
				height: parent.height
				color: logStyle.generic.layout.buttonSeparatorColor2
			}

			ToolButton {
				id: deloyButton
				checkable: true
				height: logStyle.generic.layout.headerButtonHeight
				width: 50
				anchors.verticalCenter: parent.verticalCenter
				checked: true
				onCheckedChanged: {
					proxyModel.toogleFilter("deployment")
				}
				tooltip: qsTr("Deployment")
				style:
					ButtonStyle {
					label:
						Item {
						Rectangle
						{
							width: labelDeploy.width
							height: labelDeploy.height
							anchors.centerIn: parent
							color: "transparent"
							DefaultLabel {
								width: 40
								id: labelDeploy
								anchors.horizontalCenter: parent.horizontalCenter
								elide: Text.ElideRight
								font.family: logStyle.generic.layout.logLabelFont
								font.pointSize: appStyle.absoluteSize(-3)
								color: logStyle.generic.layout.logLabelColor
								text: qsTr("Deploy.")
							}
						}
					}
					background:
						Rectangle {
						color: deloyButton.checked ? logStyle.generic.layout.buttonSelected : "transparent"
					}
				}
			}

			Rectangle {
				anchors.verticalCenter: parent.verticalCenter
				width: 1;
				height: parent.height
				color: logStyle.generic.layout.buttonSeparatorColor1
			}

			Rectangle {
				anchors.verticalCenter: parent.verticalCenter
				width: 2;
				height: parent.height
				color: logStyle.generic.layout.buttonSeparatorColor2
			}
		}

		Row
		{
			height: parent.height
			anchors.right: parent.right
			anchors.rightMargin: 10
			spacing: 10
			Rectangle
			{
				height: logStyle.generic.layout.headerButtonHeight
				anchors.verticalCenter: parent.verticalCenter
				color: "transparent"
				width: 20
				Button
				{
					id: clearButton
					action: clearAction
					anchors.fill: parent
					anchors.verticalCenter: parent.verticalCenter
					height: 25
					style:
						ButtonStyle {
						background:
							Rectangle {
							height: logStyle.generic.layout.headerButtonHeight
							implicitHeight: logStyle.generic.layout.headerButtonHeight
							color: "transparent"
						}
					}
				}

				Image {
					id: clearImage
					source: clearAction.enabled ? "qrc:/qml/img/cleariconactive.png" : "qrc:/qml/img/clearicon.png"
					anchors.centerIn: parent
					fillMode: Image.PreserveAspectFit
					width: 20
					height: 25
				}

				Action {
					id: clearAction
					enabled: logsModel.count > 0
					tooltip: qsTr("Clear")
					onTriggered: {
						logsPane.clear()
					}
				}
			}

			Rectangle
			{
				height: logStyle.generic.layout.headerButtonHeight
				anchors.verticalCenter: parent.verticalCenter
				color: "transparent"
				width: 20
				Button
				{
					id: copyButton
					action: copyAction
					anchors.fill: parent
					anchors.verticalCenter: parent.verticalCenter
					height: 25
					style:
						ButtonStyle {
						background:
							Rectangle {
							height: logStyle.generic.layout.headerButtonHeight
							implicitHeight: logStyle.generic.layout.headerButtonHeight
							color: "transparent"
						}
					}
				}

				Image {
					id: copyImage
					source: copyAction.enabled ? "qrc:/qml/img/copyiconactive.png" : "qrc:/qml/img/copyicon.png"
					anchors.centerIn: parent
					fillMode: Image.PreserveAspectFit
					width: 20
					height: 25
				}

				Action {
					id: copyAction
					enabled: logsModel.count > 0
					tooltip: qsTr("Copy to Clipboard")
					onTriggered: {
						var content = "";
						for (var k = 0; k < logsModel.count; k++)
						{
							var log = logsModel.get(k);
							content += log.type + "\t" + log.level + "\t" + log.date + "\t" + log.content + "\n";
						}
						clipboard.text = content;
					}
				}
			}

			Rectangle
			{
				width: 120
				radius: 10
				height: 25
				color: "white"
				anchors.verticalCenter: parent.verticalCenter

				Image
				{
					id: searchImg
					source: "qrc:/qml/img/searchicon.png"
					fillMode: Image.PreserveAspectFit
					width: 20
					height: 25
					z: 3
				}

				DefaultTextField
				{
					id: searchBox
					z: 2
					width: 100
					anchors.left: searchImg.right
					anchors.leftMargin: -7
					font.family: logStyle.generic.layout.logLabelFont
					font.pointSize: appStyle.absoluteSize(-3)
					font.italic: true
					text: qsTr(" - Search - ")
					onFocusChanged:
					{
						if (!focus && text === "")
							text = qsTr(" - Search - ");
						else if (focus && text === qsTr(" - Search - "))
							text = "";
					}

					onTextChanged: {
						if (text === qsTr(" - Search - "))
							proxyModel.search("");
						else
							proxyModel.search(text);
					}

					style:
						TextFieldStyle {
						background: Rectangle {
							radius: 10
						}
					}
				}
			}
		}
	}

}
