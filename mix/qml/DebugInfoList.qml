import QtQuick 2.2
import QtQuick.Controls 1.1
import QtQuick.Layouts 1.0
import QtQuick.Controls.Styles 1.1

ColumnLayout {
	id: root
	property string title
	property variant listModel;
	property bool collapsible;
	property bool collapsed;
	property bool enableSelection: false;
	property real storedHeight: 0;
	property Component itemDelegate
	property Component componentDelegate
	property alias item: loader.item
	signal rowActivated(int index)
	spacing: 0

	function collapse()
	{
		storedHeight = childrenRect.height;
		storageContainer.collapse();
	}

	function show()
	{
		storageContainer.expand();
	}

	Component.onCompleted:
	{
		if (storageContainer.parent.parent.height === 25)
			storageContainer.collapse();
		else
		{
			if (storageContainer.parent.parent.height === 0)
				storageContainer.parent.parent.height = 200;
			storageContainer.expand();
		}
	}

	RowLayout {
		height: 25
		id: header
		Image {
			source: "img/closedtriangleindicator.png"
			width: 15
			height: 15
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
					if (collapsed)
					{
						storageContainer.expand();
						if (storedHeight <= 25)
							storedHeight = 200;
						storageContainer.parent.parent.height = storedHeight;
					}
					else
					{
						storedHeight = root.childrenRect.height;
						storageContainer.collapse();
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

		function collapse() {
			storageImgArrow.source = "qrc:/qml/img/closedtriangleindicator.png";
			if (storageContainer.parent.parent.height > 25)
				storageContainer.parent.parent.height = 25;
			collapsed = true;
		}

		function expand() {
			storageImgArrow.source = "qrc:/qml/img/opentriangleindicator.png";
			collapsed = false;
		}

		Loader
		{
			id: loader
			anchors.top: parent.top
			anchors.left: parent.left
			anchors.topMargin: 3
			anchors.leftMargin: 3
			width: parent.width - 3
			height: parent.height - 6
			onHeightChanged:  {
				if (height <= 0 && collapsible) {
					storageContainer.collapse();
				}
				else if (height > 0 && collapsed) {
					storageContainer.expand();
				}
			}

			sourceComponent: componentDelegate ? componentDelegate : table
		}
		Component
		{
			id: table
			TableView
			{
				clip: true;
				alternatingRowColors: false
				anchors.fill: parent
				model: listModel
				selectionMode: enableSelection ? SelectionMode.SingleSelection : SelectionMode.NoSelection
				headerDelegate: null
				itemDelegate: root.itemDelegate
				onActivated: rowActivated(row);
				Keys.onPressed: {
					if ((event.modifiers & Qt.ControlModifier) && event.key === Qt.Key_C && currentRow >=0 && currentRow < listModel.length) {
						var str = "";
						for (var i = 0; i < listModel.length; i++)
							str += listModel[i] + "\n";
						clipboard.text = str;
					}
				}

				TableViewColumn {
					role: "modelData"
					width: parent.width
				}
			}
		}
	}
}
