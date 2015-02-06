pragma Singleton
import QtQuick 2.0

/*
 * Project Files
 */
QtObject {
	property QtObject general: QtObject {
		property int leftMargin: 45
	}

	property QtObject title: QtObject {
		property string color: "#808080"
		property string background: "#f0f0f0"
		property int height: 70
		property int pointSize: 15
	}

	property QtObject documentsList: QtObject {
		property string background: "#f7f7f7"
		property string color: "#4d4d4d"
		property string sectionColor: "#808080"
		property string selectedColor: "white"
		property string highlightColor: "#4a90e2"
		property int height: 32
		property int fileNameHeight: 45
		property int fontSize: 15
	}
}
