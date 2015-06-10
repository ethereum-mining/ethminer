import QtQuick 2.2
import QtQuick.Controls 1.1
import QtQuick.Layouts 1.0
import QtQuick.Controls.Styles 1.1
import Qt.labs.settings 1.0
import org.ethereum.qml.QEther 1.0
import "js/QEtherHelper.js" as QEtherHelper
import "js/TransactionHelper.js" as TransactionHelper
import "."

Rectangle {
	objectName: "mainContent"
	signal keyPressed(variant event)
	focus: true
	Keys.enabled: true
	Keys.onPressed:
	{
		root.keyPressed(event.key);
	}
	anchors.fill: parent
	id: root

	property alias rightViewVisible: scenarioExe.visible
	property alias webViewVisible: webPreview.visible
	property alias webView: webPreview
	property alias projectViewVisible: projectList.visible
	property alias projectNavigator: projectList
	property alias runOnProjectLoad: mainSettings.runOnProjectLoad
	property alias rightPane: scenarioExe
	property alias debuggerPanel: debugPanel
	property alias codeEditor: codeEditor
	property bool webViewHorizontal: codeWebSplitter.orientation === Qt.Vertical //vertical splitter positions elements vertically, splits screen horizontally
	property bool firstCompile: true

	Connections {
		target: codeModel
		onCompilationComplete: {
			if (firstCompile) {
				firstCompile = false;
				if (runOnProjectLoad)
					startQuickDebugging();
			}
		}
	}

	Connections {
		target: debugPanel
		onDebugExecuteLocation: {
			codeEditor.highlightExecution(documentId, location);
		}
	}

	Connections {
		target: codeEditor
		onBreakpointsChanged: {
			debugPanel.setBreakpoints(codeEditor.getBreakpoints());
		}
	}

	function startQuickDebugging()
	{
		ensureRightView();
		projectModel.stateListModel.debugDefaultState();
	}

	function toggleRightView() {
		scenarioExe.visible = !scenarioExe.visible;
	}

	function ensureRightView() {
		scenarioExe.visible = true;
	}

	function rightViewIsVisible()
	{
		return scenarioExe.visible;
	}

	function hideRightView() {
		scenarioExe.visible = lfalse;
	}

	function toggleWebPreview() {
		webPreview.visible = !webPreview.visible;
	}

	function toggleProjectView() {
		projectList.visible = !projectList.visible;
	}

	function toggleWebPreviewOrientation() {
		codeWebSplitter.orientation = (codeWebSplitter.orientation === Qt.Vertical ? Qt.Horizontal : Qt.Vertical);
	}

	//TODO: move this to debugger.js after refactoring, introduce events
	function toggleBreakpoint() {
		codeEditor.toggleBreakpoint();
	}

	function displayCompilationErrorIfAny()
	{
		scenarioExe.visible = true;
		scenarioExe.displayCompilationErrorIfAny();
	}

	Settings {
		id: mainSettings
		property alias codeWebOrientation: codeWebSplitter.orientation
		property alias webWidth: webPreview.width
		property alias webHeight: webPreview.height
		property alias showProjectView: projectList.visible
		property bool runOnProjectLoad: true
	}

	ColumnLayout
	{
		id: mainColumn
		anchors.fill: parent
		spacing: 0
		Rectangle {
			width: parent.width
			height: 50
			Layout.row: 0
			Layout.fillWidth: true
			Layout.preferredHeight: 50
			id: headerView
			Rectangle
			{
				gradient: Gradient {
					GradientStop { position: 0.0; color: "#f1f1f1" }
					GradientStop { position: 1.0; color: "#d9d7da" }
				}
				id: headerPaneContainer
				anchors.fill: parent
				StatusPane
				{
					anchors.fill: parent
					webPreview: webPreview
				}
			}
		}

		Rectangle {
			Layout.fillWidth: true
			height: 1
			color: "#8c8c8c"
		}

		Rectangle {
			Layout.fillWidth: true
			Layout.preferredHeight: root.height - headerView.height;

			Settings {
				id: splitSettings
				property alias projectWidth: projectList.width
				property alias contentViewWidth: contentView.width
				property alias rightViewWidth: scenarioExe.width
			}

			Splitter
			{
				anchors.fill: parent
				orientation: Qt.Horizontal

				ProjectList	{
					id: projectList
					width: 350
					Layout.minimumWidth: 250
					Layout.fillHeight: true
					Connections {
						target: projectModel.codeEditor
					}
				}

				Rectangle {
					id: contentView
					Layout.fillHeight: true
					Layout.fillWidth: true

					Splitter {
						id: codeWebSplitter
						anchors.fill: parent
						orientation: Qt.Vertical
						CodeEditorView {
							id: codeEditor
							height: parent.height * 0.6
							anchors.top: parent.top
							Layout.fillWidth: true
							Layout.fillHeight: true
						}
						WebPreview {
							id: webPreview
							height: parent.height * 0.4
							Layout.fillWidth: codeWebSplitter.orientation === Qt.Vertical
							Layout.fillHeight: codeWebSplitter.orientation === Qt.Horizontal
							Layout.minimumHeight: 200
							Layout.minimumWidth: 200
						}
					}
				}

				ScenarioExecution
				{
					id: scenarioExe;
					visible: false;
					Layout.fillHeight: true
					Keys.onEscapePressed: visible = false
					Layout.minimumWidth: 650
					anchors.right: parent.right
				}

				Debugger
				{
					id: debugPanel
					visible: false
					Layout.fillHeight: true
					Keys.onEscapePressed: visible = false
					Layout.minimumWidth: 650
					anchors.right: parent.right
				}

				Connections {
					target: clientModel
					onDebugDataReady:  {
						scenarioExe.visible = false
						debugPanel.visible = true
						if (scenarioExe.bc.debugTrRequested)
						{
							debugPanel.setTr(scenarioExe.bc.model.blocks[scenarioExe.bc.debugTrRequested[0]].transactions[scenarioExe.bc.debugTrRequested[1]])
						}
					}
				}

				Connections {
					target: debugPanel
					onPanelClosed:  {
						debugPanel.visible = false
						scenarioExe.visible = true
					}
				}
			}
		}
	}
}
