import QtQuick 2.2
import QtQuick.Controls 1.1
import QtQuick.Layouts 1.0
import QtQuick.Controls.Styles 1.1

Item {
  id: button

  signal clicked
  signal pressed
  signal released

  width: sprite.width
  height: sprite.height


  MouseArea {
	id: mouseArea
	enabled: button.enabled
	anchors.fill: button
	hoverEnabled: true

	onClicked: button.clicked()
	onPressed: button.pressed()
	onReleased: button.released()
  }

  onClicked: {
  }

  onPressed: {
	opacity = 0.5
  }

  onReleased: {
	opacity = 1.0
  }
}
