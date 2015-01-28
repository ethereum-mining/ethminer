import QtQuick 2.2
import QtQuick.Controls 1.1
import QtQuick.Controls.Styles 1.1
import QtQuick.Dialogs 1.1
import QtQuick.Layouts 1.1
import QtQuick.Window 2.1
import Qt.labs.settings 1.0
import org.ethereum.qml.QEther 1.0

ApplicationWindow {
	id: mainApplication
	visible: true
	width: 1200
	height: 800
	minimumWidth: 400
	minimumHeight: 300
	title: qsTr("mix")

	menuBar: MenuBar {
		Menu {
			title: qsTr("File")
			MenuItem { action: createProjectAction }
			MenuItem { action: openProjectAction }
			MenuSeparator {}
			MenuItem { action: saveAllFilesAction }
			MenuSeparator {}
			MenuItem { action: addExistingFileAction }
			MenuItem { action: addNewJsFileAction }
			MenuItem { action: addNewHtmlFileAction }
			MenuSeparator {}
			//MenuItem { action: addNewContractAction }
			MenuItem { action: closeProjectAction }
			MenuSeparator {}
			MenuItem { action: exitAppAction }
		}
		Menu {
			title: qsTr("Debug")
			MenuItem { action: debugRunAction }
			MenuItem { action: debugResetStateAction }
			MenuItem { action: mineAction }
		}
		Menu {
			title: qsTr("Windows")
			MenuItem { action: openNextDocumentAction }
			MenuItem { action: openPrevDocumentAction }
			MenuSeparator {}
			MenuItem { action: showHideRightPanelAction }
			MenuItem { action: toggleWebPreviewAction }
			MenuItem { action: toggleWebPreviewOrientationAction }
		}
	}

	MainContent {
		id: mainContent;
		anchors.fill: parent
	}

	ModalDialog {
		objectName: "dialog"
		id: dialog
	}

	AlertMessageDialog {
		objectName: "alertMessageDialog"
		id: messageDialog
	}

	Settings {
		id: mainWindowSettings
		property alias mainWidth: mainApplication.width
		property alias mainHeight: mainApplication.height
		property alias mainX: mainApplication.x
		property alias mainY: mainApplication.y
	}

	Action {
		id: exitAppAction
		text: qsTr("Exit")
		shortcut: "Ctrl+Q"
		onTriggered: Qt.quit();
	}

	Action {
		id: mineAction
		text: "Mine"
		shortcut: "Ctrl+M"
		onTriggered: clientModel.mine();
		enabled: codeModel.hasContract && !clientModel.running
	}
	Action {
		id: debugRunAction
		text: "&Run"
		shortcut: "F5"
		onTriggered: mainContent.startQuickDebugging()
		enabled: codeModel.hasContract && !clientModel.running
	}

	Action {
		id: debugResetStateAction
		text: "Reset &State"
		shortcut: "F6"
		onTriggered: clientModel.resetState();
	}

	Action {
		id: toggleWebPreviewAction
		text: "Show Web View"
		shortcut: "F2"
		checkable: true
		checked: mainContent.webViewVisible
		onTriggered: mainContent.toggleWebPreview();
	}

	Action {
		id: toggleWebPreviewOrientationAction
		text: "Horizontal Web View"
		shortcut: ""
		checkable: true
		checked: mainContent.webViewHorizontal
		onTriggered: mainContent.toggleWebPreviewOrientation();
	}

	Action {
		id: showHideRightPanelAction
		text: "Show Right View"
		shortcut: "F7"
		checkable: true
		checked: mainContent.rightViewVisible
		onTriggered: mainContent.toggleRightView();
	}

	Action {
		id: createProjectAction
		text: qsTr("&New Project")
		shortcut: "Ctrl+N"
		enabled: true;
		onTriggered: projectModel.createProject();
	}

	Action {
		id: openProjectAction
		text: qsTr("&Open Project")
		shortcut: "Ctrl+O"
		enabled: true;
		onTriggered: projectModel.browseProject();
	}

	Action {
		id: addNewJsFileAction
		text: qsTr("New JavaScript File")
		shortcut: "Ctrl+Alt+J"
		enabled: !projectModel.isEmpty
		onTriggered: projectModel.newJsFile();
	}

	Action {
		id: addNewHtmlFileAction
		text: qsTr("New HTML File")
		shortcut: "Ctrl+Alt+H"
		enabled: !projectModel.isEmpty
		onTriggered: projectModel.newHtmlFile();
	}

	Action {
		id: addNewContractAction
		text: qsTr("New Contract")
		shortcut: "Ctrl+Alt+C"
		enabled: !projectModel.isEmpty
		onTriggered: projectModel.newContract();
	}

	Action {
		id: addExistingFileAction
		text: qsTr("Add Existing File")
		shortcut: "Ctrl+Alt+A"
		enabled: !projectModel.isEmpty
		onTriggered: projectModel.addExistingFile();
	}

	Action {
		id: saveAllFilesAction
		text: qsTr("Save All")
		shortcut: "Ctrl+S"
		enabled: !projectModel.isEmpty
		onTriggered: projectModel.saveAll();
	}

	Action {
		id: closeProjectAction
		text: qsTr("Close Project")
		shortcut: "Ctrl+W"
		enabled: !projectModel.isEmpty
		onTriggered: projectModel.closeProject();
	}

	Action {
		id: openNextDocumentAction
		text: qsTr("Next Document")
		shortcut: "Ctrl+Tab"
		enabled: !projectModel.isEmpty
		onTriggered: projectModel.openNextDocument();
	}

	Action {
		id: openPrevDocumentAction
		text: qsTr("Previous Document")
		shortcut: "Ctrl+Shift+Tab"
		enabled: !projectModel.isEmpty
		onTriggered: projectModel.openPrevDocument();
	}

}
