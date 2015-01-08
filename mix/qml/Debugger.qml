import QtQuick 2.2
import QtQuick.Controls.Styles 1.1
import QtQuick.Controls 1.1
import QtQuick.Dialogs 1.1
import QtQuick.Layouts 1.1
import "js/Debugger.js" as Debugger
import "js/ErrorLocationFormater.js" as ErrorLocationFormater

Rectangle {
	id: debugPanel
	objectName: "debugPanel"
	anchors.fill: parent;
	color: "#ededed"

	Keys.onPressed:
	{
		if (event.key === Qt.Key_F10)
			Debugger.moveSelection(1);
		else if (event.key === Qt.Key_F9)
			Debugger.moveSelection(-1);
	}

	function init()
	{
		if (constantCompilation.result.successfull)
		{
			Debugger.init();
			debugScrollArea.visible = true;
			compilationErrorArea.visible = false;
			machineStates.visible = true;
		}
		else
		{
			debugScrollArea.visible = false;
			compilationErrorArea.visible = true;
			machineStates.visible = false;
			console.log(constantCompilation.result.compilerMessage);
			var errorInfo = ErrorLocationFormater.extractErrorInfo(constantCompilation.result.compilerMessage, false);
			errorLocation.text = errorInfo.errorLocation;
			errorDetail.text = errorInfo.errorDetail;
			errorLine.text = errorInfo.errorLine;
		}
		forceActiveFocus();
	}

	Rectangle
	{
		visible: false;
		id: compilationErrorArea
		width: parent.width - 20
		height: 500
		color: "#ededed"
		anchors.left: parent.left
		anchors.top: parent.top
		anchors.margins: 10
		ColumnLayout
		{
			width: parent.width
			anchors.top: parent.top
			spacing: 25
			RowLayout
			{
				height: 100
				Image {
					id: compileFailed
					source: "qrc:/qml/img/compilfailed.png"
				}
				ColumnLayout
				{
					Text {
						color: "red"
						id: errorLocation
					}
					Text {
						color: "#4a4a4a"
						id: errorDetail
					}
				}
			}

			Rectangle
			{
				width: parent.width - 6
				height: 2
				color: "#d0d0d0"
			}

			RowLayout
			{
				Text
				{
					color: "#4a4a4a"
					id: errorLine
				}
			}
		}
	}


	Flickable {
		id: debugScrollArea
		flickableDirection: Flickable.VerticalFlick
		anchors.fill: parent
		contentHeight: machineStates.height
		contentWidth: machineStates.width

		GridLayout
		{
			property int sideMargin: 10
			id: machineStates
			anchors.top: parent.top
			anchors.topMargin: 15
			anchors.left: parent.left;
			anchors.leftMargin: machineStates.sideMargin
			anchors.right: parent.right;
			anchors.rightMargin: machineStates.sideMargin
			flow: GridLayout.TopToBottom
			//columnSpacing: 7
			rowSpacing: 15
			RowLayout {
				// step button + slider
				spacing: 10
				height: 27
				width: debugPanel.width
				RowLayout {
					id: jumpButtons
					spacing: 15
					width: 250
					height: parent.height

					StepActionImage
					{
						id: jumpoutbackaction;
						source: "qrc:/qml/img/jumpoutback.png"
						disableStateImg: "qrc:/qml/img/jumpoutbackdisabled.png"
						onClicked: Debugger.stepOutBack()
					}

					StepActionImage
					{
						id: jumpintobackaction
						source: "qrc:/qml/img/jumpintoback.png"
						disableStateImg: "qrc:/qml/img/jumpintobackdisabled.png"
						onClicked: Debugger.stepIntoBack()
					}

					StepActionImage
					{
						id: jumpoverbackaction
						source: "qrc:/qml/img/jumpoverback.png"
						disableStateImg: "qrc:/qml/img/jumpoverbackdisabled.png"
						onClicked: Debugger.stepOverBack()
					}

					StepActionImage
					{
						id: jumpoverforwardaction
						source: "qrc:/qml/img/jumpoverforward.png"
						disableStateImg: "qrc:/qml/img/jumpoverforwarddisabled.png"
						onClicked: Debugger.stepOverForward()
					}

					StepActionImage
					{
						id: jumpintoforwardaction
						source: "qrc:/qml/img/jumpintoforward.png"
						disableStateImg: "qrc:/qml/img/jumpintoforwarddisabled.png"
						onClicked: Debugger.stepIntoForward()
					}

					StepActionImage
					{
						id: jumpoutforwardaction
						source: "qrc:/qml/img/jumpoutforward.png"
						disableStateImg: "qrc:/qml/img/jumpoutforwarddisabled.png"
						onClicked: Debugger.stepOutForward()
					}
				}

				Rectangle {
					color: "transparent"
					width: 250
					height: parent.height
					Slider {
						id: statesSlider
						anchors.fill: parent
						tickmarksEnabled: true
						stepSize: 1.0
						height: parent.height
						onValueChanged: Debugger.jumpTo(value);
						style: SliderStyle {
							groove: Rectangle {
								implicitHeight: 3
								color: "#7da4cd"
								radius: 8
							}
							handle: Rectangle {
								anchors.centerIn: parent
								color: control.pressed ? "white" : "lightgray"
								border.color: "gray"
								border.width: 2
								implicitWidth: 10
								implicitHeight: 10
								radius: 12
							}
						}
					}
				}
			}

			RowLayout {
				// Assembly code
				width: debugPanel.width
				height: 400
				spacing: 10

				Rectangle
				{
					width: 170
					height: parent.height
					border.width: 3
					border.color: "#deddd9"
					color: "white"

					ListView {
						anchors.fill: parent
						anchors.topMargin: 3
						anchors.bottomMargin: 3
						clip: true
						id: statesList
						delegate: renderDelegate
						highlight: highlightBar
						highlightFollowsCurrentItem: true
					}

					Component {
						id: highlightBar
						Rectangle {
							height: statesList.currentItem.height
							width: statesList.currentItem.width
							color: "#4b8fe2"
							Behavior on y { SpringAnimation { spring: 2; damping: 0.1 } }
						}
					}

					Component {
						id: renderDelegate
						Item {
							id: wrapperItem
							height: 20
							width: parent.width
							Text {
								color: parent.ListView.isCurrentItem ? "white" : "black"
								anchors.centerIn: parent
								text: line
								font.pointSize: 9
							}
						}
					}
				}

				ColumnLayout {
					width: 250
					height: parent.height
					Rectangle {
						// Info
						width: parent.width
						id: basicInfoColumn
						height: 150
						color: "transparent"
						DebugBasicInfo {
							id: basicInfo
							width: parent.width
							height: parent.height
						}
					}

					Rectangle {
						// Stack
						height: 250
						width: parent.width
						color: "transparent"

						Storage {
							id: stack
							width: parent.width
							title : qsTr("Stack")
						}
					}
				}
			}

			Rectangle {
				width: debugPanel.width - 2 * machineStates.sideMargin
				height: 2;
				color: "#e3e3e3"
				radius: 3
			}

			Storage {
				id: storage
				width: debugPanel.width - 2 * machineStates.sideMargin
				title : qsTr("Storage")
			}

			Rectangle {
				width: debugPanel.width - 2 * machineStates.sideMargin
				height: 2;
				color: "#e3e3e3"
				radius: 3
			}

			Storage {
				id: memoryDump
				width: debugPanel.width - 2 * machineStates.sideMargin
				title: qsTr("Memory Dump")
			}

			Rectangle {
				width: debugPanel.width - 2 * machineStates.sideMargin
				height: 2;
				color: "#e3e3e3"
				radius: 3
			}

			Storage {
				id: callDataDump
				width: debugPanel.width;
				title: qsTr("Call data")
			}
		}
	}
}
