import QtQuick 2.0
import QtQuick.Window 2.0
import QtQuick.Layouts 1.0
import QtQuick.Controls 1.0
import org.ethereum.qml.ProjectModel 1.0

Item {

	ColumnLayout {
		anchors.fill: parent
		Text {
			Layout.fillWidth: true
			color: "blue"
			text: ProjectModel.projectData ? ProjectModel.projectData.title : ""
			horizontalAlignment: Text.AlignHCenter
			visible: !ProjectModel.isEmpty;
		}
		ListView {
			Layout.fillWidth: true
			Layout.fillHeight: true
			id: projectList
			model: ProjectModel.listModel

			delegate: renderDelegate
			highlight: Rectangle {
				color: "lightsteelblue";
			}
			highlightFollowsCurrentItem: true
			focus: true
			clip: true

			onCurrentIndexChanged: {
				if (currentIndex >= 0 && currentIndex < ProjectModel.listModel.count)
					ProjectModel.openDocument(ProjectModel.listModel.get(currentIndex).documentId);
			}
		}
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
					text: name
					font.pointSize: 12
					verticalAlignment: Text.AlignBottom
				}
			}
			MouseArea {
				id: mouseArea
				z: 1
				hoverEnabled: false
				anchors.fill: parent

				onClicked:{
					projectList.currentIndex = index;
				}
			}

			Connections {
				target: ProjectModel
				onProjectLoaded: {
					projectList.currentIndex = 0;
				}
			}
		}
	}
}

