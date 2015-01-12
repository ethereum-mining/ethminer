import QtQuick 2.2
import QtQuick.Controls 1.1
import QtQuick.Controls.Styles 1.1
import QtQuick.Dialogs 1.1
import QtQuick.Layouts 1.1
import QtQuick.Window 2.1
import CodeEditorExtensionManager 1.0
import org.ethereum.qml.ProjectModel 1.0

ApplicationWindow {
	id: mainApplication
	visible: true
	width: 1200
	height: 600
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
		}
	}
	Component.onCompleted: {
		setX(Screen.width / 2 - width / 2);
		setY(Screen.height / 2 - height / 2);
	}

	MainContent {
	}

	ModalDialog {
		objectName: "dialog"
		id: dialog
	}

	AlertMessageDialog {
		objectName: "alertMessageDialog"
		id: messageDialog
	}

	Action {
		id: exitAppAction
		text: qsTr("Exit")
		shortcut: "Ctrl+Q"
		onTriggered: Qt.quit();
	}

	Action {
		id: debugRunAction
		text: "&Run"
		shortcut: "F5"
		enabled: codeModel.hasContract && !clientModel.running;
		onTriggered: clientModel.debugDeployment();
	}

	Action {
		id: debugResetStateAction
		text: "Reset &State"
		shortcut: "F6"
		onTriggered: clientModel.resetState();
	}

	Action {
		id: createProjectAction
		text: qsTr("&New project")
		shortcut: "Ctrl+N"
		enabled: true;
		onTriggered: ProjectModel.createProject();
	}

	Action {
		id: openProjectAction
		text: qsTr("&Open project")
		shortcut: "Ctrl+O"
		enabled: true;
		onTriggered: ProjectModel.browseProject();
	}

	Action {
		id: addNewJsFileAction
		text: qsTr("New JavaScript file")
		shortcut: "Ctrl+Alt+J"
		enabled: !ProjectModel.isEmpty
		onTriggered: ProjectModel.newJsFile();
	}

	Action {
		id: addNewHtmlFileAction
		text: qsTr("New HTML file")
		shortcut: "Ctrl+Alt+H"
		enabled: !ProjectModel.isEmpty
		onTriggered: ProjectModel.newHtmlFile();
	}

	Action {
		id: addNewContractAction
		text: qsTr("New contract")
		shortcut: "Ctrl+Alt+C"
		enabled: !ProjectModel.isEmpty
		onTriggered: ProjectModel.newContract();
	}

	Action {
		id: addExistingFileAction
		text: qsTr("Add existing file")
		shortcut: "Ctrl+Alt+A"
		enabled: !ProjectModel.isEmpty
		onTriggered: ProjectModel.addExistingFile();
	}

	Action {
		id: saveAllFilesAction
		text: qsTr("Save all")
		shortcut: "Ctrl+S"
		enabled: !ProjectModel.isEmpty
		onTriggered: ProjectModel.saveAll();
	}

	Action {
		id: closeProjectAction
		text: qsTr("Close project")
		shortcut: "Ctrl+W"
		enabled: !ProjectModel.isEmpty
		onTriggered: ProjectModel.closeProject();
	}
}
