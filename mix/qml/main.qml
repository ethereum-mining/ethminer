import QtQuick 2.2
import QtQuick.Controls 1.1
import QtQuick.Controls.Styles 1.1
import QtQuick.Dialogs 1.1
import QtQuick.Layouts 1.1
import QtQuick.Window 2.0
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
}
