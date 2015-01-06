import QtQuick 2.2
import QtQuick.Controls 1.2
import QtQuick.Controls.Styles 1.2
import QtQuick.Dialogs 1.1
import QtQuick.Layouts 1.1
import QtQuick.Window 2.1
import CodeEditorExtensionManager 1.0

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


}
