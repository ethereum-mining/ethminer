import QtQuick 2.2
import QtQuick.Controls 1.1
import QtQuick.Layouts 1.0
import QtQuick.Controls.Styles 1.1


ColumnLayout {
	property variant listModel;
	property string title;
	height: 250
	//width: width //debugPanel.width
	//anchors.left: machineStates.left
	//anchors.right: machineStates.right
	//anchors.leftMargin: machineStates.sideMargin


	RowLayout {

		Image {
			source: "qrc:/qml/img/jumpoverback.png"
			width: 15
			sourceSize.width: 15
			id: imgArrow

		}

		Text {
			color: "#8b8b8b"
			text: title
			id: listTitle
		}

		MouseArea
		{
			anchors.fill: parent
			onClicked: {
				if (listContainer.state === "collapsed")
					listContainer.state = "";
				else
					listContainer.state = "collapsed";
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
					target: listContainer
					height: 0
					opacity: 0
					visible: false
				}
				PropertyChanges {
					target: listContainer.parent
					height: 20
				}
				PropertyChanges {
					target: dumpList
					height: 0
					opacity: 0
					visible: false
				}

			}
		]
		id: listContainer
		border.width: 3
		border.color: "#deddd9"
		anchors.top: listTitle.bottom
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
			id: dumpList
			delegate: Component {
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
