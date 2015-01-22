import QtQuick 2.2
import QtQuick.Controls 1.1
import QtQuick.Controls.Styles 1.1
import QtQuick.Dialogs 1.1
import QtQuick.Layouts 1.1
import QtQuick.Window 2.1
import CodeEditorExtensionManager 1.0
import org.ethereum.qml.QEther 1.0
import "js/QEtherHelper.js" as QEtherHelper
import "js/TransactionHelper.js" as TransactionHelper

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
		Menu {
			title: qsTr("Windows")
			MenuItem { action: showHideRightPanel }
		}
	}

	Component.onCompleted: {
		setX(Screen.width / 2 - width / 2);
		setY(Screen.height / 2 - height / 2);
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
		onTriggered: {
			var item = TransactionHelper.defaultTransaction();
			item.executeConstructor = true;
			if (codeModel.code.contract.constructor.parameters.length === 0)
			{
				mainContent.ensureRightView();
				startF5Debugging(item);
			}
			else
				transactionDialog.open(0, item);
		}
		enabled: codeModel.hasContract && !clientModel.running;
	}

	function startF5Debugging(transaction)
	{
		var ether = QEtherHelper.createEther("100000000000000000000000000", QEther.Wei);
		var state = {
			title: "",
			balance: ether,
			transactions: [transaction]
		};
		clientModel.debugState(state);
	}

	TransactionDialog {
		id: transactionDialog
		onAccepted: {
			mainContent.ensureRightView();
			var item = transactionDialog.getItem();
			item.executeConstructor = true;
			startF5Debugging(item);
		}
		useTransactionDefaultValue: true
	}

	Action {
		id: debugResetStateAction
		text: "Reset &State"
		shortcut: "F6"
		onTriggered: clientModel.resetState();
	}

	Action {
		id: showHideRightPanel
		text: "Show/Hide right view"
		shortcut: "F7"
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
}
