import QtQuick 2.0
import QtQuick.Window 2.0
import QtQuick.Layouts 1.0
import QtQuick.Controls 1.0
import org.ethereum.qml.ProjectModel 1.0

Item {
	ListView {
		model: ProjectModel.listModel
		delegate: renderDelegate
	}

	Component {
		id: renderDelegate
		Item {
			id: wrapperItem
			height: 20
			width: parent.width
			RowLayout {
				anchors.fill: parent
				Text {
					Layout.fillWidth: true
					Layout.fillHeight: true
					text: title
					font.pointSize: 12
					verticalAlignment: Text.AlignBottom
				}
			}
		}
	}

	Action {
		id: createProjectAction
		text: qsTr("&New project")
		shortcut: "Ctrl+N"
		enabled: true;
		onTriggered: ProjectModel.createProject();
	}
}
