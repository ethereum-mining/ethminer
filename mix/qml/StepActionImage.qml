import QtQuick 2.2
import QtQuick.Controls 1.1
import QtQuick.Layouts 1.0
import QtQuick.Controls.Styles 1.1

ColumnLayout {
	property string title
	property variant listModel;
	height: 250

	RowLayout {

		Image {
			source: "qrc:/qml/img/jumpoverback.png"
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
			anchors.fill: parent
			onClicked: {
				if (storageContainer.state == "collapsed")
					storageContainer.state = "";
				else
					storageContainer.state = "collapsed";
			}
		}
	}

	Rectangle
	{
		Layout.fillWidth: true
		states: [
			State {
				name: "collapsed"
				PropertyChanges {
					target: storageContainer
					height: 0
					opacity: 0
					visible: false
				}
				PropertyChanges {
					target: storageContainer.parent
					height: 20
				}
				PropertyChanges {
					target: storageList
					height: 0
					opacity: 0
					visible: false
				}

			}
		]
		id: storageContainer
		border.width: 3
		border.color: "#deddd9"
		anchors.top: storageListTitle.bottom
		height: 223
		anchors.topMargin: 5
		width: parent.width
		ListView {
			anchors.top: parent.top
			anchors.left: parent.left
			anchors.topMargin: 5
			anchors.leftMargin: 5
			width: parent.width
			height: parent.height
			id: storageList
			model: listModel
			delegate:
				Component {
								Item {
									height: 20
									width: parent.width
									Text {
										color: "#8b8b8b"
										text: modelData
										font.pointSize: 9
									}
								}
							}
		}
	}
}
