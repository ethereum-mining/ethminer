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

	property alias debugSlider: statesSlider
	property alias solLocals: solLocals
	property alias solStorage: solStorage
	property alias solCallStack: solCallStack
	property alias vmCallStack: callStack
	property alias vmStorage: storage
	property alias vmMemory: memoryDump
	property alias vmCallData: callDataDump
	signal debugExecuteLocation(string documentId, var location)
	property string compilationErrorMessage
	property bool assemblyMode: false
	signal panelClosed
	objectName: "debugPanel"
	color: "#ededed"
	clip: true

	onVisibleChanged:
	{
		if (visible)
			forceActiveFocus();
	}

	onAssemblyModeChanged:
	{
		Debugger.updateMode();
		machineStates.updateHeight();
	}

	function setTr(tr)
	{
		trName.text = tr.label
	}

	function displayCompilationErrorIfAny()
	{
		debugScrollArea.visible = false;
		compilationErrorArea.visible = true;
		machineStates.visible = false;
		var errorInfo = ErrorLocationFormater.extractErrorInfo(compilationErrorMessage, false);
		errorLocation.text = errorInfo.errorLocation;
		errorDetail.text = errorInfo.errorDetail;
		errorLine.text = errorInfo.line;
	}

	function update(data, giveFocus)
	{
		if (data === null)
			Debugger.init(null);
		else if (data.states.length === 0)
			Debugger.init(null);
		else if (codeModel.hasContract)
		{
			Debugger.init(data);
			debugScrollArea.visible = true;
			machineStates.visible = true;
		}
		if (giveFocus)
			forceActiveFocus();
	}

	function setBreakpoints(bp)
	{
		Debugger.setBreakpoints(bp);
	}

	DebuggerPaneStyle {
		id: dbgStyle
	}

	Connections {
		target: clientModel
		onDebugDataReady:  {
			update(_debugData, false);
		}
	}

	Connections {
		target: codeModel
		onCompilationComplete: {
			debugPanel.compilationErrorMessage = "";
		}

		onCompilationError: {
			debugPanel.compilationErrorMessage = _error;
		}
	}

	Settings {
		id: splitSettings
		property alias callStackHeight: callStackRect.height
		property alias storageHeightSettings: storageRect.height
		property alias memoryDumpHeightSettings: memoryRect.height
		property alias callDataHeightSettings: callDataRect.height
		property alias solCallStackHeightSettings: solStackRect.height
		property alias solStorageHeightSettings: solStorageRect.height
		property alias solLocalsHeightSettings: solLocalsRect.height
	}

	ColumnLayout {
		id: debugScrollArea
		anchors.fill: parent
		spacing: 0
		RowLayout
		{
			Layout.preferredWidth: parent.width
			Layout.preferredHeight: 30
			Rectangle
			{
				Layout.preferredWidth: parent.width
				Layout.preferredHeight: parent.height
				color: "transparent"
				Text {
					anchors.centerIn: parent
					text: qsTr("Current Transaction")
				}

				Rectangle
				{
					anchors.left: parent.left
					anchors.leftMargin: 10
					width: 30
					height: parent.height
					color: "transparent"
					anchors.verticalCenter: parent.verticalCenter
					Image {
						source: "qrc:/qml/img/leftarrow@2x.png"
						width: parent.width
						fillMode: Image.PreserveAspectFit
						anchors.centerIn: parent
					}
					MouseArea
					{
						anchors.fill: parent
						onClicked:
						{
							Debugger.init(null);
							panelClosed()
						}
					}
				}
			}
		}

		RowLayout
		{
			Layout.preferredWidth: parent.width
			Layout.preferredHeight: 30
			Rectangle
			{
				Layout.preferredWidth: parent.width
				Layout.preferredHeight: parent.height
				color: "#2C79D3"
				Text {
					id: trName
					color: "white"
					anchors.centerIn: parent
				}
			}
		}

		ScrollView
		{
			property int sideMargin: 10
			id: machineStates
			Layout.fillWidth: true
			Layout.fillHeight: true
			function updateHeight() {
				var h = buttonRow.childrenRect.height;
				if (assemblyMode)
					h += assemblyCodeRow.childrenRect.height + callStackRect.childrenRect.height + storageRect.childrenRect.height + memoryRect.childrenRect.height + callDataRect.childrenRect.height;
				else
					h += solStackRect.childrenRect.height + solLocalsRect.childrenRect.height + solStorageRect.childrenRect.height;
				statesLayout.height = h + 120;
			}

			Component.onCompleted: updateHeight();

			ColumnLayout {
				id: statesLayout
				anchors.top: parent.top
				anchors.topMargin: 15
				anchors.left: parent.left;
				anchors.leftMargin: machineStates.sideMargin
				width: debugScrollArea.width - machineStates.sideMargin * 2 - 20
				spacing: machineStates.sideMargin

				Rectangle {
					// step button + slider
					id: buttonRow
					height: 30
					Layout.fillWidth: true
					color: "transparent"

					Rectangle {
						anchors.fill: parent
						color: "transparent"
						RowLayout {
							anchors.fill: parent
							id: jumpButtons
							spacing: 3
							layoutDirection: Qt.LeftToRight

							StepActionImage
							{
								id: runBackAction;
								enabledStateImg: "qrc:/qml/img/jumpoutback.png"
								disableStateImg: "qrc:/qml/img/jumpoutbackdisabled.png"
								onClicked: Debugger.runBack()
								width: 23
								buttonShortcut: "Ctrl+Shift+F5"
								buttonTooltip: qsTr("Run Back")
								visible: false
							}

							StepActionImage
							{
								id: jumpOutBackAction;
								enabledStateImg: "qrc:/qml/img/jumpoutback.png"
								disableStateImg: "qrc:/qml/img/jumpoutbackdisabled.png"
								onClicked: Debugger.stepOutBack()
								width: 23
								buttonShortcut: "Ctrl+Shift+F11"
								buttonTooltip: qsTr("Step Out Back")
							}

							StepActionImage
							{
								id: jumpIntoBackAction
								enabledStateImg: "qrc:/qml/img/jumpintoback.png"
								disableStateImg: "qrc:/qml/img/jumpintobackdisabled.png"
								onClicked: Debugger.stepIntoBack()
								width: 23
								buttonShortcut: "Ctrl+F11"
								buttonTooltip: qsTr("Step Into Back")
							}

							StepActionImage
							{
								id: jumpOverBackAction
								enabledStateImg: "qrc:/qml/img/jumpoverback.png"
								disableStateImg: "qrc:/qml/img/jumpoverbackdisabled.png"
								onClicked: Debugger.stepOverBack()
								width: 23
								buttonShortcut: "Ctrl+F10"
								buttonTooltip: qsTr("Step Over Back")
							}

							StepActionImage
							{
								id: jumpOverForwardAction
								enabledStateImg: "qrc:/qml/img/jumpoverforward.png"
								disableStateImg: "qrc:/qml/img/jumpoverforwarddisabled.png"
								onClicked: Debugger.stepOverForward()
								width: 23
								buttonShortcut: "F10"
								buttonTooltip: qsTr("Step Over Forward")
							}

							StepActionImage
							{
								id: jumpIntoForwardAction
								enabledStateImg: "qrc:/qml/img/jumpintoforward.png"
								disableStateImg: "qrc:/qml/img/jumpintoforwarddisabled.png"
								onClicked: Debugger.stepIntoForward()
								width: 23
								buttonShortcut: "F11"
								buttonTooltip: qsTr("Step Into Forward")
							}

							StepActionImage
							{
								id: jumpOutForwardAction
								enabledStateImg: "qrc:/qml/img/jumpoutforward.png"
								disableStateImg: "qrc:/qml/img/jumpoutforwarddisabled.png"
								onClicked: Debugger.stepOutForward()
								width: 45
								buttonShortcut: "Shift+F11"
								buttonTooltip: qsTr("Step Out Forward")
								buttonRight: true
							}

							StepActionImage
							{
								id: runForwardAction
								enabledStateImg: "qrc:/qml/img/jumpoutforward.png"
								disableStateImg: "qrc:/qml/img/jumpoutforwarddisabled.png"
								onClicked: Debugger.runForward()
								width: 45
								buttonShortcut: "Ctrl+F5"
								buttonTooltip: qsTr("Run Forward")
								visible: false
								buttonRight: true
							}

							Rectangle {
								anchors.top: parent.top
								anchors.bottom: parent.bottom
								anchors.right: parent.right
								color: "transparent"
								Layout.fillWidth: true
								Layout.minimumWidth: parent.width * 0.2
								Layout.alignment: Qt.AlignRight

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
					}
				}

				Rectangle {
					// Assembly code
					id: assemblyCodeRow
					Layout.fillWidth: true
					height: 405
					implicitHeight: 405
					color: "transparent"
					visible: assemblyMode

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
										font.pointSize: dbgStyle.general.basicFontSize
										wrapMode: Text.NoWrap
										id: id
									}
									Text {
										anchors.left: id.right;
										wrapMode: Text.NoWrap
										color: styleData.selected ? "white" : "black"
										font.family: "monospace"
										text: styleData.value.replace(styleData.value.split(' ')[0], '')
										font.pointSize: dbgStyle.general.basicFontSize
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
						height: parent.height
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
												font.pointSize: dbgStyle.general.basicFontSize
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
												font.pointSize: dbgStyle.general.basicFontSize
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
						id: solStackRect;
						color: "transparent"
						Layout.minimumHeight: 25
						Layout.maximumHeight: 800
						onHeightChanged: machineStates.updateHeight();
						visible: !assemblyMode
						CallStack {
							anchors.fill: parent
							id: solCallStack
						}
					}

					Rectangle
					{
						id: solLocalsRect;
						color: "transparent"
						Layout.minimumHeight: 25
						Layout.maximumHeight: 800
						onHeightChanged: machineStates.updateHeight();
						visible: !assemblyMode
						VariablesView {
							title : qsTr("Locals")
							anchors.fill: parent
							id: solLocals
						}
					}

					Rectangle
					{
						id: solStorageRect;
						color: "transparent"
						Layout.minimumHeight: 25
						Layout.maximumHeight: 800
						onHeightChanged: machineStates.updateHeight();
						visible: !assemblyMode
						VariablesView {
							title : qsTr("Members")
							anchors.fill: parent
							id: solStorage
						}
					}

					Rectangle
					{
						id: callStackRect;
						color: "transparent"
						Layout.minimumHeight: 25
						Layout.maximumHeight: 800
						onHeightChanged: machineStates.updateHeight();
						visible: assemblyMode
						CallStack {
							anchors.fill: parent
							id: callStack
							onRowActivated: Debugger.displayFrame(index);
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
						visible: assemblyMode
						StorageView {
							anchors.fill: parent
							id: storage
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
						visible: assemblyMode
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
						visible: assemblyMode
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
