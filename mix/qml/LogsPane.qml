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
		logsModel.insert(0, { "type": _type, "date": Qt.formatDateTime(new Date(), "hh:mm:ss dd.MM.yyyy"), "content": _content, "level": _level });
	}

	anchors.fill: parent
	radius: 5
	color: "#f7f7f7"
	border.color: "#808080"
	border.width: 1
	ColumnLayout {
		height: parent.height
		width: parent.width
		spacing: 0
		Row
		{
			id: rowAction
			Layout.preferredHeight: 35
			height: 35
			anchors.leftMargin: 10
			anchors.left: parent.left
			spacing: 5
			Button
			{
				height: 30
				anchors.verticalCenter: parent.verticalCenter
				action: clearAction
				iconSource: "qrc:/qml/img/broom.png"
			}

			Action {
				id: clearAction
				enabled: logsModel.count > 0
				tooltip: qsTr("Clear")
				onTriggered: {
					logsModel.clear()
				}
			}

			Button
			{
				height: 30
				anchors.verticalCenter: parent.verticalCenter
				action: copytoClipBoardAction
				iconSource: "qrc:/qml/img/copy.png"
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
					appContext.toClipboard(content);
				}
			}

			Rectangle {
				anchors.verticalCenter: parent.verticalCenter
				width: 1;
				height: parent.height - 10
				color : "#808080"
			}

			ToolButton {
				id: javascriptButton
				checkable: true
				height: 30
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
							font.family: "sans serif"
							font.pointSize: Style.absoluteSize(-3)
							color: "#5391d8"
							anchors.centerIn: parent
							text: qsTr("JavaScript")
						}
					}
			}
		}

		ToolButton {
			id: runButton
			checkable: true
			height: 30
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
						font.family: "sans serif"
						font.pointSize: Style.absoluteSize(-3)
						color: "#5391d8"
						anchors.centerIn: parent
						text: qsTr("Run")
					}
				}
			}
		}

		ToolButton {
			id: stateButton
			checkable: true
			height: 30
			anchors.verticalCenter: parent.verticalCenter
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
						font.family: "sans serif"
						font.pointSize: Style.absoluteSize(-3)
						color: "#5391d8"
						anchors.centerIn: parent
						text: qsTr("State")
					}
				}
			}
		}

		DefaultTextField
		{
			id: searchBox
			height: 30
			anchors.verticalCenter: parent.verticalCenter
			width: 200
			font.family: "sans serif"
			font.pointSize: Style.absoluteSize(-3)
			onTextChanged: {
				proxyModel.search(text);
			}
		}
	}

	ListModel {
		id: logsModel

	}

	TableView {
		id: logsTable
		clip: true
		Layout.fillWidth: true
		Layout.preferredHeight: parent.height - rowAction.height
		headerVisible : false
		onDoubleClicked:
		{
			var log = logsModel.get((logsTable.currentRow));
			appContext.toClipboard(log.type + " " + log.level + " " + log.date + " " + log.content);
		}

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
				console.log("(?:" + roles.join('|') + ")");
				return "(?:" + roles.join('|') + ")";
			}
			filterType: "(?:javascript|run|state)"
			filterContent: ""
			filterSyntax: SortFilterProxyModel.RegExp
			filterCaseSensitivity: Qt.CaseInsensitive
		}
		TableViewColumn
		{
			role: "date"
			title: qsTr("date")
			width: 150
			delegate: itemDelegate
		}
		TableViewColumn
		{
			role: "type"
			title: qsTr("type")
			width: 100
			delegate: itemDelegate
		}
		TableViewColumn
		{
			role: "content"
			title: qsTr("content")
			width: 700
			delegate: itemDelegate
		}
	}

	Component {
		id: itemDelegate
		DefaultLabel {
			text: styleData.value;
			font.family: "sans serif"
			font.pointSize: Style.absoluteSize(-1)
			color: "#808080"
		}
	}
}
}
