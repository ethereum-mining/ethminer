import QtQuick 2.0
import QtQuick.Window 2.0
import QtQuick.Layouts 1.0
import QtQuick.Controls 1.0
import org.ethereum.qml.ProjectModel 1.0

Item {
	ListView {
		id: projectList
		model: ProjectModel.listModel
		anchors.fill: parent
		delegate: renderDelegate
		highlight: Rectangle {
			color: "lightsteelblue";
		}
		highlightFollowsCurrentItem: true
		focus: true
		clip: true
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
					ProjectModel.documentOpen(ProjectModel.listModel.get(index));
				}
			}
		}
	}

}
