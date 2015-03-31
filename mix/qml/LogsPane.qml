import QtQuick 2.0
import QtQuick.Layouts 1.0
import QtQuick.Controls 1.1
import QtQuick.Controls.Styles 1.3
import org.ethereum.qml.SortFilterProxyModel 1.0
import "."

Rectangle
{
	function push(_level, _type, _content)
	{
		_content = _content.replace(/\n/g, " ")
		logsModel.insert(0, { "type": _type, "date": Qt.formatDateTime(new Date(), "hh:mm:ss"), "content": _content, "level": _level });
	}

	anchors.fill: parent
	radius: 5
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
			ColumnLayout
			{
				id: logsRect
				spacing: 0
				Repeater {
					clip: true
					model: SortFilterProxyModel {
						id: proxyModel
						source: logsModel
						property var roles: ["-", "javascript", "run", "state"]

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

						filterType: "(?:javascript|run|state)"
						filterContent: ""
						filterSyntax: SortFilterProxyModel.RegExp
						filterCaseSensitivity: Qt.CaseInsensitive
					}

					Rectangle
					{
						width: 750
						height: 30
						color:
						{
							if (level === "warning" || level === "error")
								return "#fffcd5";
							else
								return index % 2 === 0 ? "transparent" : LogsPaneStyle.generic.layout.logAlternateColor;
						}


						DefaultLabel {
							text: date;
							font.family: LogsPaneStyle.generic.layout.logLabelFont
							width: LogsPaneStyle.generic.layout.dateWidth
							font.pointSize: Style.absoluteSize(-1)
							anchors.left: parent.left
							anchors.leftMargin: 15
							anchors.verticalCenter: parent.verticalCenter
							color: {
								parent.getColor(level);
							}
						}

						DefaultLabel {
							text: type;
							font.family: LogsPaneStyle.generic.layout.logLabelFont
							width: LogsPaneStyle.generic.layout.typeWidth
							font.pointSize: Style.absoluteSize(-1)
							anchors.left: parent.left
							anchors.leftMargin: 100
							anchors.verticalCenter: parent.verticalCenter
							color: {
								parent.getColor(level);
							}
						}

						DefaultLabel {
							text: content;
							font.family: LogsPaneStyle.generic.layout.logLabelFont
							width: LogsPaneStyle.generic.layout.contentWidth
							font.pointSize: Style.absoluteSize(-1)
							anchors.verticalCenter: parent.verticalCenter
							anchors.left: parent.left
							anchors.leftMargin: 190
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
				font.family: LogsPaneStyle.generic.layout.logLabelFont
				font.pointSize: Style.absoluteSize(-1)
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
		Layout.preferredHeight: LogsPaneStyle.generic.layout.headerHeight
		height: LogsPaneStyle.generic.layout.headerHeight
		width: logsPane.width
		anchors.bottom: parent.bottom
		Row
		{
			id: rowAction
			anchors.leftMargin: LogsPaneStyle.generic.layout.leftMargin
			anchors.left: parent.left
			spacing: LogsPaneStyle.generic.layout.headerButtonSpacing
			height: parent.height
			Rectangle
			{
				color: "transparent"
				height: 20
				width: 50
				anchors.verticalCenter: parent.verticalCenter
				DefaultLabel
				{
					color: "#808080"
					font.family: LogsPaneStyle.generic.layout.logLabelFont
					text: qsTr("Show:")
				}
			}

			Rectangle {
				anchors.verticalCenter: parent.verticalCenter
				width: 1;
				height: parent.height
				color : "#808080"
			}

			ToolButton {
				id: javascriptButton
				checkable: true
				height: LogsPaneStyle.generic.layout.headerButtonHeight
				width: 20
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
						DefaultLabel {
							font.family: LogsPaneStyle.generic.layout.logLabelFont
							font.pointSize: Style.absoluteSize(-3)
							color: LogsPaneStyle.generic.layout.logLabelColor
							anchors.centerIn: parent
							text: qsTr("JS")
						}
					}
					background:
						Rectangle {
						color: javascriptButton.checked ? "#cfcfcf" : "transparent"
					}
				}
			}

			Rectangle {
				anchors.verticalCenter: parent.verticalCenter
				width: 1;
				height: parent.height
				color : "#808080"
			}

			ToolButton {
				id: runButton
				checkable: true
				height: LogsPaneStyle.generic.layout.headerButtonHeight
				width: 30
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
						DefaultLabel {
							font.family: LogsPaneStyle.generic.layout.logLabelFont
							font.pointSize: Style.absoluteSize(-3)
							color: LogsPaneStyle.generic.layout.logLabelColor
							anchors.centerIn: parent
							text: qsTr("Run")
						}
					}
					background:
						Rectangle {
						color: runButton.checked ? "#cfcfcf" : "transparent"
					}
				}
			}

			Rectangle {
				anchors.verticalCenter: parent.verticalCenter
				width: 1;
				height: parent.height
				color : "#808080"
			}

			ToolButton {
				id: stateButton
				checkable: true
				height: LogsPaneStyle.generic.layout.headerButtonHeight
				anchors.verticalCenter: parent.verticalCenter
				width: 35
				checked: true
				onCheckedChanged: {
					proxyModel.toogleFilter("state")
				}
				tooltip: qsTr("State")
				style:
					ButtonStyle {
					label:
						Item {
						DefaultLabel {
							font.family: LogsPaneStyle.generic.layout.logLabelFont
							font.pointSize: Style.absoluteSize(-3)
							color: LogsPaneStyle.generic.layout.logLabelColor
							anchors.centerIn: parent
							text: qsTr("State")
						}
					}
					background:
						Rectangle {
						color: stateButton.checked ? "#cfcfcf" : "transparent"
					}
				}
			}

			Rectangle {
				anchors.verticalCenter: parent.verticalCenter
				width: 1;
				height: parent.height
				color : "#808080"
			}
		}

		Row
		{
			height: parent.height
			anchors.right: parent.right
			anchors.rightMargin: 4
			spacing: 4
			Button
			{
				id: clearButton
				height: LogsPaneStyle.generic.layout.headerButtonHeight
				anchors.verticalCenter: parent.verticalCenter
				action: hideAction
				iconSource: "qrc:/qml/img/cleariconactive.png"
				style:
					ButtonStyle {
					background:
						Rectangle {
						height: LogsPaneStyle.generic.layout.headerButtonHeight
						implicitHeight: LogsPaneStyle.generic.layout.headerButtonHeight
						color: "transparent"
					}
				}
			}

			Image {
				id: clearImage
				source: "qrc:/qml/img/cleariconactive.png"
				anchors.centerIn: parent
				fillMode: Image.PreserveAspectFit
				width: 30
				height: 30
			}

			Button
			{
				id: exitButton
				height: LogsPaneStyle.generic.layout.headerButtonHeight
				anchors.verticalCenter: parent.verticalCenter
				action: exitAction
				iconSource: "qrc:/qml/img/exit.png"
				style:
					ButtonStyle {
					background:
						Rectangle {
						height: LogsPaneStyle.generic.layout.headerButtonHeight
						color: "transparent"
					}
				}
			}

			Button
			{
				id: copyButton
				height: LogsPaneStyle.generic.layout.headerButtonHeight
				anchors.verticalCenter: parent.verticalCenter
				action: copytoClipBoardAction
				iconSource: "qrc:/qml/img/copyiconactive.png"
				style:
					ButtonStyle {
					background:
						Rectangle {
						height: LogsPaneStyle.generic.layout.headerButtonHeight
						color: "transparent"
					}
				}
			}

			Action {
				id: clearAction
				tooltip: qsTr("Hide")
				onTriggered: {
					logsPane.parent.toggle();
				}
			}

			Action {
				id: hideAction
				enabled: logsModel.count > 0
				tooltip: qsTr("Clear")
				onTriggered: {
					logsModel.clear()
				}
			}

			Action {
				id: copytoClipBoardAction
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

			DefaultTextField
			{
				id: searchBox
				height: LogsPaneStyle.generic.layout.headerButtonHeight - 5
				anchors.verticalCenter: parent.verticalCenter
				width: LogsPaneStyle.generic.layout.headerInputWidth - 40
				font.family: LogsPaneStyle.generic.layout.logLabelFont
				font.pointSize: Style.absoluteSize(-3)
				font.italic: true
				text: qsTr("Search")
				onTextChanged: {
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
