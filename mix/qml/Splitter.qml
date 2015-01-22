import QtQuick 2.2

Rectangle {
	property variant itemToStick;
	property int itemMinimumWidth;
	property string direction;
	property variant brother;

	Component.onCompleted:
	{
		if (direction === "left")
			anchors.right = itemToStick.left;
		else if (direction === "right")
			anchors.left = itemToStick.right;
	}

	width: 5
	height: parent.height
	anchors.top: parent.top;
	MouseArea
	{
		property int startX: 0;
		anchors.fill: parent
		onPressed: startX = mouseX;
		onPositionChanged:
		{
			parent.x += mouseX;
			var diff = 0;
			if (direction == "left")
				diff = mouseX - startX;
			else if (direction == "right")
				diff = -(mouseX - startX);

			if (itemMinimumWidth > itemToStick.width - diff)
			{
				brother.width = brother.width + diff;
				itemToStick.width = itemMinimumWidth;
			}
			else
			{
				brother.width = brother.width + diff;
				itemToStick.width = itemToStick.width - diff;
			}
		}
		cursorShape: Qt.SizeHorCursor
	}
}
