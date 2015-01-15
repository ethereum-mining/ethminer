import QtQuick 2.2
import QtQuick.Controls 1.1
import QtQuick.Layouts 1.0
import QtQuick.Controls.Styles 1.1

ColumnLayout {
	property string title
	property variant listModel;
	property bool collapsible;
	property Component itemDelegate
	spacing: 0
	RowLayout {
		height: 25
		id: header
		Image {
			source: "qrc:/qml/img/opentriangleindicator.png"
			width: 15
			sourceSize.width: 15
			id: storageImgArrow
			visible: collapsible
		}

		Text {
			anchors.left: storageImgArrow.right
			color: "#8b8b8b"
			text: title
			id: storageListTitle
		}

		MouseArea
		{
			enabled: collapsible
			anchors.fill: parent
			onClicked: {
				if (storageContainer.state == "collapsed")
					storageContainer.state = "";
				else
					storageContainer.state = "collapsed";
			}
		}
	}

	RowLayout
	{
		height: parent.height - header.height
		clip: true
		Rectangle
		{
			height: parent.height
			border.width: 3
			border.color: "#deddd9"
			Layout.fillWidth: true
			states: [
				State {
					name: "collapsed"
					PropertyChanges {
						target: storageContainer.parent
						height: 0
						visible: false
					}
					PropertyChanges {
						target: storageImgArrow
						source: "qrc:/qml/img/closedtriangleindicator.png"
					}
				}
			]
			id: storageContainer
			width: parent.width
			ListView {
				clip: true;
				anchors.top: parent.top
				anchors.left: parent.left
				anchors.topMargin: 3
				anchors.leftMargin: 3
				width: parent.width - 3
				height: parent.height - 6
				id: storageList
				model: listModel
				delegate: itemDelegate
			}
		}
	}
}
