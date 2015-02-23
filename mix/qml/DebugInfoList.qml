import QtQuick 2.2
import QtQuick.Controls 1.1
import QtQuick.Layouts 1.0
import QtQuick.Controls.Styles 1.1

ColumnLayout {
	id: root
	property string title
	property variant listModel;
	property bool collapsible;
	property bool enableSelection: false;
	property real storedHeight: 0;
	property Component itemDelegate
	signal rowActivated(int index)
	spacing: 0

	function collapse()
	{
		storedHeight = childrenRect.height;
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
			height: 15
			sourceSize.width: 15
			sourceSize.height: 15
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
						storageContainer.parent.parent.height = storedHeight;
					}
					else
					{
						storedHeight = root.childrenRect.height;
						storageContainer.state = "collapsed";
					}
				}
			}
		}
	}
	Rectangle
	{
		id: storageContainer
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
		TableView {
			clip: true;
			alternatingRowColors: false
			anchors.top: parent.top
			anchors.left: parent.left
			anchors.topMargin: 3
			anchors.leftMargin: 3
			width: parent.width - 3
			height: parent.height - 6
			model: listModel
			selectionMode: enableSelection ? SelectionMode.SingleSelection : SelectionMode.NoSelection
			headerDelegate: null
			itemDelegate: root.itemDelegate
			onHeightChanged:  {
				if (height <= 0 && collapsible) {
					if (storedHeight <= 0)
						storedHeight = 200;
					storageContainer.state = "collapsed";
				}
				else if (height > 0 && storageContainer.state == "collapsed") {
					//TODO: fix increasing size
					//storageContainer.state = "";
				}
			}
			onActivated: rowActivated(row);
			Keys.onPressed: {
				if ((event.modifiers & Qt.ControlModifier) && event.key === Qt.Key_C && currentRow >=0 && currentRow < listModel.length) {
					var str = "";
					for (var i = 0; i < listModel.length; i++)
						str += listModel[i] + "\n";
					appContext.toClipboard(str);
				}
			}

			TableViewColumn {
				role: "modelData"
				width: parent.width
			}
		}
	}
}
