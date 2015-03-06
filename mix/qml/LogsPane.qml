import QtQuick 2.0
import QtQuick.Layouts 1.0
import QtQuick.Controls 1.1
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
	border.width: 2
	ColumnLayout {
		height: parent.height - 4
		width: parent.width - 2
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
						content += log.type + "\t" + log.level + "\t" + log.date + "\t" + log.content;
					}
					appContext.toClipboard(content);
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

				filterRole: ""
				filterString: ""
				filterSyntax: SortFilterProxyModel.Wildcard
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
				color: {
					if (styleData.row > -1)
					{
						var l = logsModel.get(styleData.row).level
						if (l === "error")
							return "red"
						else if (l === "warning")
							return "orange"
						else if (l === "info")
							return "#808080"
					}
					else
						return "#808080"
				}
			}
		}
	}
}
