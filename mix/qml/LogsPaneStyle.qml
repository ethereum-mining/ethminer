import QtQuick 2.0

QtObject {

	function absoluteSize(rel)
	{
		return systemPointSize + rel;
	}

	property QtObject generic: QtObject {
		property QtObject layout: QtObject {
			property string backgroundColor: "#f7f7f7"
			property int headerHeight: 30
			property int headerButtonSpacing: 0
			property int leftMargin: 10
			property int headerButtonHeight: 30
			property string logLabelColor: "#4a4a4a"
			property string logLabelFont: "sans serif"
			property int headerInputWidth: 200
			property int dateWidth: 100
			property int typeWidth: 100
			property int contentWidth: 560
			property string logAlternateColor: "#f6f5f6"
			property string errorColor: "#fffcd5"
			property string buttonSeparatorColor1: "#d3d0d0"
			property string buttonSeparatorColor2: "#f2f1f2"
			property string buttonSelected: "#dcdcdc"
		}
	}
}
