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

	function collapse()
	{
		storageContainer.state = "collapsed";
	}

	function show()
	{
		storageContainer.state = "";
	}

	Component.onCompleted:
	{
		if (storageContainer.parent.parent.height === 25)
			storageContainer.state = "collapsed";
	}

	RowLayout {
		height: 25
		id: header
		Image {
			source: "qrc:/qml/img/opentriangleindicator.png"
			width: 15
			sourceSize.width: 15
			id: storageImgArrow
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
				if (collapsible)
				{
					if (storageContainer.state == "collapsed")
					{
						storageContainer.state = "";
						storageContainer.parent.parent.height = storageContainer.parent.parent.Layout.maximumHeight;
					}
					else
						storageContainer.state = "collapsed";
				}
			}
		}
	}
	Rectangle
	{
		border.width: 3
		border.color: "#deddd9"
		Layout.fillWidth: true
		Layout.fillHeight: true
		states: [
			State {
				name: "collapsed"
				PropertyChanges {
					target: storageImgArrow
					source: "qrc:/qml/img/closedtriangleindicator.png"
				}
				PropertyChanges {
					target: storageContainer.parent.parent
					height: 25
				}
			}
		]
		id: storageContainer
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
