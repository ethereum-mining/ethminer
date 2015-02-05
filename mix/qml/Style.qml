pragma Singleton
import QtQuick 2.0

/*
 * Project Files
 */
QtObject {
	property QtObject general: QtObject {
		property int leftMargin: 30
	}

	property QtObject title: QtObject {
		property string color: "#878787"
		property string background: "#f0f0f0"
		property int height: 40
	}

	property QtObject documentsList: QtObject {
		property string background: "#f7f7f7"
		property string color: "#4d4d4d"
		property string selectedColor: "white"
		property string highlightColor: "#4a90e2"
		property int height: 30
		property int fontSize: 12
	}
}
