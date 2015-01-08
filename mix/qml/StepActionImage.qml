import QtQuick 2.2
import QtQuick.Controls 1.1
import QtQuick.Layouts 1.0
import QtQuick.Controls.Styles 1.1

Image {
	id: jumpintobackimg
	property string disableStateImg
	signal clicked
	width: 15
	sourceSize.width: 15
	MouseArea
	{
		anchors.fill: parent
		onClicked: jumpintobackimg.clicked();
	}
	states: [
		State {
			name: "disabled"
			PropertyChanges {
				target: jumpintobackimg
				source: disableStateImg
			}
		}
	]
}
