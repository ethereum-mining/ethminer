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
			MenuItem { action: addExistingFileAction }
			MenuItem { action: addNewJsFileAction }
			MenuItem { action: addNewHtmlFileAction }
			MenuItem { action: addNewContractAction }
			MenuItem {
				text: qsTr("Exit")
				onTriggered: Qt.quit();
			}
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
		id: debugRunAction
		text: "&Run"
		shortcut: "F5"
		enabled: codeModel.hasContract && !debugModel.running;
		onTriggered: debugModel.debugDeployment();
	}

	Action {
		id: debugResetStateAction
		text: "Reset &State"
		shortcut: "F6"
		onTriggered: debugModel.resetState();
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
		onTriggered: ProjectModel.addJsFile();
	}

	Action {
		id: addNewHtmlFileAction
		text: qsTr("New HTML file")
		shortcut: "Ctrl+Alt+H"
		enabled: !ProjectModel.isEmpty
		onTriggered: ProjectModel.addHtmlFile();
	}

	Action {
		id: addNewContractAction
		text: qsTr("New contract")
		shortcut: "Ctrl+Alt+C"
		enabled: !ProjectModel.isEmpty
		onTriggered: ProjectModel.addContract();
	}

	Action {
		id: addExistingFileAction
		text: qsTr("Add existing file")
		shortcut: "Ctrl+Alt+A"
		enabled: !ProjectModel.isEmpty
		onTriggered: ProjectModel.addExistingFile();
	}
}
