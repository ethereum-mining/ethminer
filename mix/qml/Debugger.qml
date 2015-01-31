import QtQuick 2.2
import QtQuick.Controls.Styles 1.1
import QtQuick.Controls 1.1
import QtQuick.Dialogs 1.1
import QtQuick.Layouts 1.1
import Qt.labs.settings 1.0
import "js/Debugger.js" as Debugger
import "js/ErrorLocationFormater.js" as ErrorLocationFormater

Rectangle {
	id: debugPanel
	objectName: "debugPanel"
	anchors.fill: parent;
	color: "#ededed"
	clip: true

	onVisibleChanged:
	{
		if (visible)
			forceActiveFocus();
	}

	function update(data, giveFocus)
	{
		if (statusPane.result.successful)
		{
			Debugger.init(data);
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
		if (giveFocus)
			forceActiveFocus();
	}

	Connections {
		target: clientModel
		onDebugDataReady:  {
			update(_debugData, true);
		}
	}

	Connections {
		target: codeModel
		onCompilationComplete: update(null, false);
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
		property int firstColumnWidth: 180
		property int secondColumnWidth: 250
		id: debugScrollArea
		flickableDirection: Flickable.VerticalFlick
		anchors.fill: parent
		contentHeight: 4000
		contentWidth: parent.width
		Rectangle
		{
			color: "transparent"
			anchors.fill: parent
			ColumnLayout
			{
				property int sideMargin: 10
				id: machineStates
				anchors.top: parent.top
				anchors.topMargin: 15
				anchors.left: parent.left;
				anchors.leftMargin: machineStates.sideMargin
				anchors.right: parent.right;
				anchors.rightMargin: machineStates.sideMargin
				anchors.fill: parent
				Layout.fillWidth: true
				Layout.fillHeight: true

				TransactionLog {
					Layout.fillWidth: true
					height: 250
				}

				RowLayout {
					// step button + slider
					id: buttonRow
					spacing: machineStates.sideMargin
					height: 27
					Layout.fillWidth: true

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
								width: 28
								height: 30
								buttonShortcut: "Ctrl+Shift+F11"
								buttonTooltip: qsTr("Step Out Back")
							}

							StepActionImage
							{
								id: jumpIntoBackAction
								enabledStateImg: "qrc:/qml/img/jumpintoback.png"
								disableStateImg: "qrc:/qml/img/jumpintobackdisabled.png"
								onClicked: Debugger.stepIntoBack()
								width: 28
								height: 30
								buttonShortcut: "Ctrl+F11"
								buttonTooltip: qsTr("Step Into Back")
							}

							StepActionImage
							{
								id: jumpOverBackAction
								enabledStateImg: "qrc:/qml/img/jumpoverback.png"
								disableStateImg: "qrc:/qml/img/jumpoverbackdisabled.png"
								onClicked: Debugger.stepOverBack()
								width: 28
								height: 30
								buttonShortcut: "Ctrl+F10"
								buttonTooltip: qsTr("Step Over Back")
							}

							StepActionImage
							{
								id: jumpOverForwardAction
								enabledStateImg: "qrc:/qml/img/jumpoverforward.png"
								disableStateImg: "qrc:/qml/img/jumpoverforwarddisabled.png"
								onClicked: Debugger.stepOverForward()
								width: 28
								height: 30
								buttonShortcut: "F10"
								buttonTooltip: qsTr("Step Over Forward")
							}

							StepActionImage
							{
								id: jumpIntoForwardAction
								enabledStateImg: "qrc:/qml/img/jumpintoforward.png"
								disableStateImg: "qrc:/qml/img/jumpintoforwarddisabled.png"
								onClicked: Debugger.stepIntoForward()
								width: 28
								height: 30
								buttonShortcut: "F11"
								buttonTooltip: qsTr("Step Into Forward")
							}

							StepActionImage
							{
								id: jumpOutForwardAction
								enabledStateImg: "qrc:/qml/img/jumpoutforward.png"
								disableStateImg: "qrc:/qml/img/jumpoutforwarddisabled.png"
								onClicked: Debugger.stepOutForward()
								width: 28
								height: 30
								buttonShortcut: "Shift+F11"
								buttonTooltip: qsTr("Step Out Forward")
							}
						}
					}

					Rectangle {
						color: "transparent"
						Layout.fillWidth: true
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
					id: assemblyCodeRow
					Layout.fillWidth: true
					height: 405
					implicitHeight: 405
					spacing: machineStates.sideMargin

					Rectangle
					{
						id: stateListContainer
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
							//highlightFollowsCurrentItem: false
							model: ListModel {}
						}

						Component {
							id: highlightBar
							Rectangle {
								radius: 4
								height: statesList.currentItem.height
								width: statesList.currentItem.width;
								y: statesList.currentItem.y
								color: "#4A90E2"
								//Behavior on y {
								//	 PropertyAnimation { properties: "y"; easing.type: Easing.InOutQuad; duration: 50}
								//}
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
									font.family: "monospace"
									font.pointSize: 9
									id: id
									wrapMode: Text.NoWrap
								}
								Text {
									wrapMode: Text.NoWrap
									color: parent.ListView.isCurrentItem ? "white" : "black"
									font.family: "monospace"
									text: line.replace(line.split(' ')[0], '')
									anchors.left: id.right
									font.pointSize: 9
								}
							}
						}
					}

					Rectangle {
						Layout.fillWidth: true
						height: parent.height //- 2 * stateListContainer.border.width
						color: "transparent"
						ColumnLayout
						{
							width: parent.width
							anchors.fill: parent
							spacing: 0
							DebugBasicInfo {
								id: currentStep
								titleStr: qsTr("Current Step")
								Layout.fillWidth: true
								height: 30
							}
							DebugBasicInfo {
								id: mem
								titleStr: qsTr("Adding Memory")
								Layout.fillWidth: true
								height: 30
							}
							DebugBasicInfo {
								id: stepCost
								titleStr: qsTr("Step Cost")
								Layout.fillWidth: true
								height: 30
							}
							DebugBasicInfo {
								id: gasSpent
								titleStr: qsTr("Total Gas Spent")
								Layout.fillWidth: true
								height: 30
							}
							DebugInfoList
							{
								Layout.fillHeight: true
								Layout.fillWidth: true
								id: stack
								collapsible: false
								title : qsTr("Stack")
								itemDelegate: Item {
									id: renderedItem
									height: 25
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
												font.family: "monospace"
												color: "#4a4a4a"
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
												font.family: "monospace"
												anchors.verticalCenter: parent.verticalCenter
												color: "#4a4a4a"
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

				SplitView
				{
					id: splitInfoList
					Layout.fillHeight: true
					Layout.fillWidth: true

					Settings {
						id: splitSettings
						property alias storageHeightSettings: storageRect.height
						property alias memoryDumpHeightSettings: memoryRect.height
						property alias callDataHeightSettings: callDataRect.height
					}

					orientation: Qt.Vertical
					width: debugPanel.width - 2 * machineStates.sideMargin


					Rectangle
					{
						id: callStackRect;
						color: "transparent"
						height: 120
						width: parent.width
						Layout.minimumHeight: 120
						Layout.maximumHeight: 400
						CallStack {
							anchors.fill: parent
							id: callStack
							onFrameActivated: Debugger.displayFrame(index);
						}
					}


					Rectangle
					{
						id: storageRect
						color: "transparent"
						width: parent.width
						Layout.minimumHeight: 25
						Layout.maximumHeight: 223
						height: 25
						DebugInfoList
						{
							id: storage
							anchors.fill: parent
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
											font.family: "monospace"
											anchors.leftMargin: 5
											color: "#4a4a4a"
											text: modelData.split('\t')[0];
											font.pointSize: 9
											width: parent.width - 5
											elide: Text.ElideRight
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
											font.family: "monospace"
											anchors.verticalCenter: parent.verticalCenter
											color: "#4a4a4a"
											text: modelData.split('\t')[1];
											elide: Text.ElideRight
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
					}

					Rectangle
					{
						id: memoryRect;
						color: "transparent"
						height: 25
						width: parent.width
						Layout.minimumHeight: 25
						Layout.maximumHeight: 223
						DebugInfoList {
							id: memoryDump
							anchors.fill: parent
							collapsible: true
							title: qsTr("Memory Dump")
							itemDelegate:
								Item {
								height: 29
								width: parent.width - 3;
								ItemDelegateDataDump {}
							}
						}
					}

					Rectangle
					{
						id: callDataRect
						color: "transparent"
						height: 25
						width: parent.width
						Layout.minimumHeight: 25
						Layout.maximumHeight: 223
						DebugInfoList {
							id: callDataDump
							anchors.fill: parent
							collapsible: true
							title: qsTr("Call Data")
							itemDelegate:
								Item {
								height: 29
								width: parent.width - 3;
								ItemDelegateDataDump {}
							}
						}
					}
					Rectangle
					{
						width: parent.width
						Layout.minimumHeight: 25
						color: "transparent"
					}
				}
			}
		}
	}
}
