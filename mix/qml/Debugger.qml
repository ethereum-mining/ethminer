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
	clip: true
	Keys.onPressed:
	{
		if (event.key === Qt.Key_F10)
			Debugger.moveSelection(1);
		else if (event.key === Qt.Key_F9)
			Debugger.moveSelection(-1);
	}

	function update()
	{
		if (statusPane.result.successful)
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
			var errorInfo = ErrorLocationFormater.extractErrorInfo(statusPane.result.compilerMessage, false);
			errorLocation.text = errorInfo.errorLocation;
			errorDetail.text = errorInfo.errorDetail;
			errorLine.text = errorInfo.errorLine;
		}
	}

	Connections {
		target: codeModel
		onCompilationComplete: update()
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
		property int firstColumnWidth: 170
		property int secondColumnWidth: 250
		id: debugScrollArea
		flickableDirection: Flickable.VerticalFlick
		anchors.fill: parent
		contentHeight: machineStates.height + 300
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
			rowSpacing: 15
			RowLayout {
				// step button + slider
				spacing: machineStates.sideMargin
				height: 27
				width: debugPanel.width
				Rectangle
				{
					height: parent.height
					color: "transparent"
					width: debugScrollArea.firstColumnWidth
					RowLayout {
						anchors.horizontalCenter: parent.horizontalCenter
						id: jumpButtons
						spacing: 3
						StepActionImage
						{
							id: jumpOutBackAction;
							enabledStateImg: "qrc:/qml/img/jumpoutback.png"
							disableStateImg: "qrc:/qml/img/jumpoutbackdisabled.png"
							onClicked: Debugger.stepOutBack()
							width: 25
							height: 27
						}

						StepActionImage
						{
							id: jumpIntoBackAction
							enabledStateImg: "qrc:/qml/img/jumpintoback.png"
							disableStateImg: "qrc:/qml/img/jumpintobackdisabled.png"
							onClicked: Debugger.stepIntoBack()
							width: 25
							height: 27
						}

						StepActionImage
						{
							id: jumpOverBackAction
							enabledStateImg: "qrc:/qml/img/jumpoverback.png"
							disableStateImg: "qrc:/qml/img/jumpoverbackdisabled.png"
							onClicked: Debugger.stepOverBack()
							width: 25
							height: 27
						}

						StepActionImage
						{
							id: jumpOverForwardAction
							enabledStateImg: "qrc:/qml/img/jumpoverforward.png"
							disableStateImg: "qrc:/qml/img/jumpoverforwarddisabled.png"
							onClicked: Debugger.stepOverForward()
							width: 25
							height: 27
						}

						StepActionImage
						{
							id: jumpIntoForwardAction
							enabledStateImg: "qrc:/qml/img/jumpintoforward.png"
							disableStateImg: "qrc:/qml/img/jumpintoforwarddisabled.png"
							onClicked: Debugger.stepIntoForward()
							width: 25
							height: 27
						}

						StepActionImage
						{
							id: jumpOutForwardAction
							enabledStateImg: "qrc:/qml/img/jumpoutforward.png"
							disableStateImg: "qrc:/qml/img/jumpoutforwarddisabled.png"
							onClicked: Debugger.stepOutForward()
							width: 25
							height: 27
						}
					}
				}
				Rectangle {
					color: "transparent"
					width: debugScrollArea.secondColumnWidth
					height: parent.height
					Slider {
						id: statesSlider
						anchors.fill: parent
						tickmarksEnabled: true
						stepSize: 1.0
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
				height: 405
				spacing: machineStates.sideMargin

				Rectangle
				{
					width: debugScrollArea.firstColumnWidth
					height: parent.height
					border.width: 3
					border.color: "#deddd9"
					color: "white"
					anchors.top: parent.top
					ListView {
						anchors.fill: parent
						anchors.leftMargin: 3
						anchors.rightMargin: 3
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
							radius: 4
							height: statesList.currentItem.height
							width: statesList.currentItem.width;
							color: "#4A90E2"
							Behavior on y { SpringAnimation { spring: 2; damping: 0.1 } }
						}
					}

					Component {
						id: renderDelegate
						RowLayout {
							id: wrapperItem
							height: 20
							width: parent.width
							spacing: 5
							Text {
								anchors.left: parent.left
								anchors.leftMargin: 10
								width: 15
								color: "#b2b3ae"
								text: line.split(' ')[0]
								font.pointSize: 9
								id: id
								wrapMode: Text.NoWrap
							}
							Text {
								wrapMode: Text.NoWrap
								color: parent.ListView.isCurrentItem ? "white" : "black"
								text: line.replace(line.split(' ')[0], '')
								anchors.left: id.right
								font.pointSize: 9
							}
						}
					}
				}

				ColumnLayout {
					width: debugScrollArea.secondColumnWidth
					height: parent.height
					Rectangle {
						// Info
						width: parent.width
						id: basicInfoColumn
						height: 125
						color: "transparent"
						ColumnLayout {
							spacing: 0
							width: parent.width
							height: parent.height
							DebugBasicInfo {
								id: currentStep
								titleStr: qsTr("Current step")
							}
							DebugBasicInfo {
								id: mem
								titleStr: qsTr("Adding memory")
							}
							DebugBasicInfo {
								id: stepCost
								titleStr: qsTr("Step cost")
							}
							DebugBasicInfo {
								id: gasSpent
								titleStr: qsTr("Total gas spent")
							}
						}
					}

					Rectangle {
						// Stack
						height: 275
						width: parent.width
						color: "transparent"

						DebugInfoList
						{
							id: stack
							width: parent.width
							height: parent.height
							collapsible: false
							title : qsTr("Stack")
							itemDelegate: Item {
								id: renderedItem
								height: 27
								width: parent.width
								RowLayout
								{
									anchors.fill: parent
									Rectangle
									{
										id: indexColumn
										color: "#f7f7f7"
										Layout.fillWidth: true
										Layout.minimumWidth: 30
										Layout.preferredWidth: 30
										Layout.maximumWidth: 30
										Layout.minimumHeight: parent.height
										Text {
											anchors.centerIn: parent
											anchors.leftMargin: 5
											color: "#8b8b8b"
											text: model.index;
											font.pointSize: 9
										}
									}

									Rectangle
									{
										anchors.left: indexColumn.right
										Layout.fillWidth: true
										Layout.minimumWidth: 15
										Layout.preferredWidth: 15
										Layout.maximumWidth: 60
										Layout.minimumHeight: parent.height
										Text {
											anchors.left: parent.left
											anchors.leftMargin: 5
											anchors.verticalCenter: parent.verticalCenter
											color: "#8b8b8b"
											text: modelData
											font.pointSize: 9
										}
									}
								}

								Rectangle {
								   id: separator
								   width: parent.width;
								   height: 1;
								   color: "#cccccc"
								   anchors.bottom: parent.bottom
								 }
							}
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

			DebugInfoList
			{
				id: storage
				width: debugPanel.width - 2 * machineStates.sideMargin
				height: 223
				collapsible: true
				title : qsTr("Storage")
				itemDelegate:
					Item {
					height: 27
					width: parent.width;
					RowLayout
					{
						id: row
						width: parent.width
						height: 26
						Rectangle
						{
							color: "#f7f7f7"
							Layout.fillWidth: true
							Layout.minimumWidth: parent.width / 2
							Layout.preferredWidth: parent.width / 2
							Layout.maximumWidth: parent.width / 2
							Layout.minimumHeight: parent.height
							Layout.maximumHeight: parent.height
							Text {
								anchors.verticalCenter: parent.verticalCenter
								anchors.left: parent.left
								anchors.leftMargin: 5
								color: "#8b8b8b"
								text: modelData.split(' ')[0].substring(0, 10);
								font.pointSize: 9
							}
						}
						Rectangle
						{
							color: "transparent"
							Layout.fillWidth: true
							Layout.minimumWidth: parent.width / 2
							Layout.preferredWidth: parent.width / 2
							Layout.maximumWidth: parent.width / 2
							Layout.minimumHeight: parent.height
							Layout.maximumHeight: parent.height
							Text {
								anchors.leftMargin: 5
								width: parent.width - 5
								wrapMode: Text.Wrap
								anchors.left: parent.left
								anchors.verticalCenter: parent.verticalCenter
								color: "#8b8b8b"
								text: modelData.split(' ')[1].substring(0, 10);
								font.pointSize: 9
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

			Rectangle {
				width: debugPanel.width - 2 * machineStates.sideMargin
				height: 2;
				color: "#e3e3e3"
				radius: 3
			}

			DebugInfoList {
				id: memoryDump
				width: debugPanel.width - 2 * machineStates.sideMargin
				height: 223
				collapsible: true
				title: qsTr("Memory Dump")
				itemDelegate:
					Item {
					height: 29
					width: parent.width - 3;
					ItemDelegateDataDump {}
				}
			}

			Rectangle {
				width: debugPanel.width - 2 * machineStates.sideMargin
				height: 2;
				color: "#e3e3e3"
				radius: 3
			}

			DebugInfoList {
				id: callDataDump
				width: debugPanel.width - 2 * machineStates.sideMargin
				height: 223
				collapsible: true
				title: qsTr("Call data")
				itemDelegate:
					Item {
					height: 29
					width: parent.width - 3;
					ItemDelegateDataDump {}
				}
			}
		}
	}
}
