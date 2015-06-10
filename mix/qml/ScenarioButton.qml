import QtQuick 2.2
import QtQuick.Controls 1.1
import QtQuick.Layouts 1.0
import QtQuick.Controls.Styles 1.1

Rectangle {
	id: buttonActionContainer
	property string text
	property string buttonShortcut
	property string sourceImg
	property string fillColor
	signal clicked

	Rectangle {
		id: contentRectangle
		anchors.fill: parent
		border.color: "#cccccc"
		border.width: 1
		radius: 4
		color: parent.fillColor ? parent.fillColor : "white"
		Image {
			id: debugImage
			anchors {
				left: parent.left
				right: parent.right
				top: parent.top
				bottom: parent.bottom
				bottomMargin: debugImg.pressed ? 0 : 2;
				topMargin: debugImg.pressed ? 2 : 0;
			}
			source: sourceImg
			fillMode: Image.PreserveAspectFit
			height: 30
		}

		Button {
			anchors.fill: parent
			id: debugImg
			action: buttonAction
			style: ButtonStyle {
				background: Rectangle {
					color: "transparent"
				}
			}
		}

		Action {
			id: buttonAction
			shortcut: buttonShortcut
			onTriggered: {
				buttonActionContainer.clicked();
			}
		}
	}

	Rectangle
	{
		anchors.top: contentRectangle.bottom
		anchors.topMargin: 15
		width: parent.width
		Text
		{
			text: buttonActionContainer.text
			anchors.centerIn: parent
		}
	}
}
