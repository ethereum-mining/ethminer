import QtQuick 2.2
import QtQuick.Controls 1.1
import QtQuick.Controls.Styles 1.1
import QtQuick.Dialogs 1.1
import QtQuick.Layouts 1.1
import Qt.labs.settings 1.0
import "js/Debugger.js" as Debugger
import "js/ErrorLocationFormater.js" as ErrorLocationFormater
import "."

Rectangle {
	id: debugPanel

	property alias transactionLog : transactionLog

	objectName: "debugPanel"
	color: "#ededed"
	clip: true

	onVisibleChanged:
	{
		if (visible)
			forceActiveFocus();
	}

	function update(data, giveFocus)
	{
		if (statusPane && statusPane.result.successful)
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

	Settings {
		id: splitSettings
		property alias transactionLogHeight: transactionLog.height
		property alias callStackHeight: callStackRect.height
		property alias storageHeightSettings: storageRect.height
		property alias memoryDumpHeightSettings: memoryRect.height
		property alias callDataHeightSettings: callDataRect.height
		property alias transactionLogVisible: transactionLog.visible
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

	SplitView {
		id: debugScrollArea
		anchors.fill: parent
		orientation: Qt.Vertical
		handleDelegate: Rectangle {
			height: machineStates.sideMargin
			color: "transparent"
		}

		TransactionLog {
			id: transactionLog
			Layout.fillWidth: true
			Layout.minimumHeight: 60
			height: 250
		}
		ScrollView
		{
			property int sideMargin: 10
			id: machineStates
			Layout.fillWidth: true
			Layout.fillHeight: true
			function updateHeight() {
				statesLayout.height = buttonRow.childrenRect.height + assemblyCodeRow.childrenRect.height +
						callStackRect.childrenRect.height + storageRect.childrenRect.height + memoryRect.childrenRect.height + callDataRect.childrenRect.height + 120;
			}

			Component.onCompleted: updateHeight();

			ColumnLayout {
				id: statesLayout
				anchors.top: parent.top
				anchors.topMargin: 15
				anchors.left: parent.left;
				anchors.leftMargin: machineStates.sideMargin
				width: debugScrollArea.width - machineStates.sideMargin * 2 - 20;
				spacing: machineStates.sideMargin

				Rectangle {
					// step button + slider
					id: buttonRow
					height: 27
					Layout.fillWidth: true
					color: "transparent"

					Rectangle {
						anchors.top: parent.top
						anchors.bottom: parent.bottom
						anchors.left: parent.left
						color: "transparent"
						width: stateListContainer.width
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
						anchors.top: parent.top
						anchors.bottom: parent.bottom
						anchors.right: parent.right
						width: debugInfoContainer.width
						color: "transparent"
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

				Rectangle {
					// Assembly code
					id: assemblyCodeRow
					Layout.fillWidth: true
					height: 405
					implicitHeight: 405
					color: "transparent"

					Rectangle
					{
						id: stateListContainer
						anchors.top : parent.top
						anchors.bottom: parent.bottom
						anchors.left: parent.left
						width: parent.width * 0.4
						height: parent.height
						border.width: 3
						border.color: "#deddd9"
						color: "white"
						TableView {
							id: statesList
							anchors.fill: parent
							anchors.leftMargin: 3
							anchors.rightMargin: 3
							anchors.topMargin: 3
							anchors.bottomMargin: 3
							clip: true
							headerDelegate: null
							itemDelegate: renderDelegate
							model: ListModel {}
							TableViewColumn {
								role: "line"
								width: parent.width - 10
							}

						}

						Component {
							id: highlightBar
							Rectangle {
								radius: 4
								anchors.fill: parent
								y: statesList.currentItem.y
								color: "#4A90E2"
							}
						}

						Component {
							id: renderDelegate
							Item {
								Rectangle {
									radius: 4
									anchors.fill: parent
									color: "#4A90E2"
									visible: styleData.selected;
								}

								RowLayout {
									id: wrapperItem
									anchors.fill: parent
									spacing: 5


									Text {
										anchors.left: parent.left
										anchors.leftMargin: 10
										width: 15
										color: "#b2b3ae"
										text: styleData.value.split(' ')[0]
										font.family: "monospace"
										font.pointSize: DebuggerPaneStyle.general.basicFontSize
										wrapMode: Text.NoWrap
										id: id
									}
									Text {
										anchors.left: id.right;
										wrapMode: Text.NoWrap
										color: styleData.selected ? "white" : "black"
										font.family: "monospace"
										text: styleData.value.replace(styleData.value.split(' ')[0], '')
										font.pointSize: DebuggerPaneStyle.general.basicFontSize
									}
								}
							}
						}
					}

					Rectangle {
						id: debugInfoContainer
						width: parent.width * 0.6 - machineStates.sideMargin
						anchors.top : parent.top
						anchors.bottom: parent.bottom
						anchors.right: parent.right
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
									//height: 25
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
												text: styleData.row;
												font.pointSize: DebuggerPaneStyle.general.basicFontSize
											}
										}

										Rectangle
										{
											anchors.left: indexColumn.right
											Layout.fillWidth: true
											Layout.minimumWidth: 15
											Layout.preferredWidth: 15
											Layout.minimumHeight: parent.height
											Text {
												anchors.left: parent.left
												anchors.leftMargin: 5
												font.family: "monospace"
												anchors.verticalCenter: parent.verticalCenter
												color: "#4a4a4a"
												text: styleData.value
												font.pointSize: DebuggerPaneStyle.general.basicFontSize
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
					orientation: Qt.Vertical

					Rectangle
					{
						id: callStackRect;
						color: "transparent"
						Layout.minimumHeight: 25
						Layout.maximumHeight: 800
						onHeightChanged: machineStates.updateHeight();
						DebugInfoList
						{
							id: callStack
							collapsible: true
							anchors.fill: parent
							title : qsTr("Call Stack")
							enableSelection: true
							onRowActivated: Debugger.displayFrame(index);
							itemDelegate:
								Item {
								anchors.fill: parent

								Rectangle {
									anchors.fill: parent
									color: "#4A90E2"
									visible: styleData.selected;
								}

								RowLayout
								{
									id: row
									anchors.fill: parent
									Rectangle
									{
										color: "#f7f7f7"
										Layout.fillWidth: true
										Layout.minimumWidth: 30
										Layout.maximumWidth: 30
										Text {
											anchors.verticalCenter: parent.verticalCenter
											anchors.left: parent.left
											font.family: "monospace"
											anchors.leftMargin: 5
											color: "#4a4a4a"
											text: styleData.row;
											font.pointSize: DebuggerPaneStyle.general.basicFontSize
											width: parent.width - 5
											elide: Text.ElideRight
										}
									}
									Rectangle
									{
										color: "transparent"
										Layout.fillWidth: true
										Layout.minimumWidth: parent.width - 30
										Layout.maximumWidth: parent.width - 30
										Text {
											anchors.leftMargin: 5
											width: parent.width - 5
											wrapMode: Text.NoWrap
											anchors.left: parent.left
											font.family: "monospace"
											anchors.verticalCenter: parent.verticalCenter
											color: "#4a4a4a"
											text: styleData.value;
											elide: Text.ElideRight
											font.pointSize: DebuggerPaneStyle.general.basicFontSize
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
						id: storageRect
						color: "transparent"
						width: parent.width
						Layout.minimumHeight: 25
						Layout.maximumHeight: 800
						onHeightChanged: machineStates.updateHeight();
						DebugInfoList
						{
							id: storage
							anchors.fill: parent
							collapsible: true
							title : qsTr("Storage")
							itemDelegate:
								Item {
								anchors.fill: parent
								RowLayout
								{
									id: row
									anchors.fill: parent
									Rectangle
									{
										color: "#f7f7f7"
										Layout.fillWidth: true
										Layout.minimumWidth: parent.width / 2
										Layout.maximumWidth: parent.width / 2
										Text {
											anchors.verticalCenter: parent.verticalCenter
											anchors.left: parent.left
											font.family: "monospace"
											anchors.leftMargin: 5
											color: "#4a4a4a"
											text: styleData.value.split('\t')[0];
											font.pointSize: DebuggerPaneStyle.general.basicFontSize
											width: parent.width - 5
											elide: Text.ElideRight
										}
									}
									Rectangle
									{
										color: "transparent"
										Layout.fillWidth: true
										Layout.minimumWidth: parent.width / 2
										Layout.maximumWidth: parent.width / 2
										Text {
											anchors.leftMargin: 5
											width: parent.width - 5
											wrapMode: Text.NoWrap
											anchors.left: parent.left
											font.family: "monospace"
											anchors.verticalCenter: parent.verticalCenter
											color: "#4a4a4a"
											text: styleData.value.split('\t')[1];
											elide: Text.ElideRight
											font.pointSize: DebuggerPaneStyle.general.basicFontSize
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
						width: parent.width
						Layout.minimumHeight: 25
						Layout.maximumHeight: 800
						onHeightChanged: machineStates.updateHeight();
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
						width: parent.width
						Layout.minimumHeight: 25
						Layout.maximumHeight: 800
						onHeightChanged: machineStates.updateHeight();
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
						id: bottomRect;
						width: parent.width
						Layout.minimumHeight: 20
						color: "transparent"
					}
				}
			}
		}
	}
}
