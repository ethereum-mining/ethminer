import QtQuick 2.2
import QtQuick.Controls 1.1
import QtQuick.Controls.Styles 1.1
import QtQuick.Layouts 1.1
import "."

DebugInfoList
{
	id: storage
	collapsible: true
	title : qsTr("Storage")
	itemDelegate:
		Item {
		anchors.fill: parent
		RowLayout
		{
			id: row
			anchors.fill: parent
			Rectangle
			{
				color: "#f7f7f7"
				Layout.fillWidth: true
				Layout.minimumWidth: parent.width / 2
				Layout.maximumWidth: parent.width / 2
				Text {
					anchors.verticalCenter: parent.verticalCenter
					anchors.left: parent.left
					font.family: "monospace"
					anchors.leftMargin: 5
					color: "#4a4a4a"
					text: styleData.value.split('\t')[0];
					font.pointSize: dbgStyle.general.basicFontSize
					width: parent.width - 5
					elide: Text.ElideRight
				}
			}
			Rectangle
			{
				color: "transparent"
				Layout.fillWidth: true
				Layout.minimumWidth: parent.width / 2
				Layout.maximumWidth: parent.width / 2
				Text {
					maximumLineCount: 1
					clip: true
					anchors.leftMargin: 5
					width: parent.width - 5
					wrapMode: Text.WrapAnywhere
					anchors.left: parent.left
					font.family: "monospace"
					anchors.verticalCenter: parent.verticalCenter
					color: "#4a4a4a"
					text: styleData.value.split('\t')[1];
					elide: Text.ElideRight
					font.pointSize: dbgStyle.general.basicFontSize
				}
			}
		}

		Rectangle {
			anchors.top: row.bottom
			width: parent.width;
			height: 1;
			color: "#cccccc"
			anchors.bottom: parent.bottom
		}
	}
}

