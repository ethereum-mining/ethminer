import QtQuick 2.0

QtObject {

	function absoluteSize(rel)
	{
		return systemPointSize + rel;
	}

	property QtObject general: QtObject {
		property string headerBackgroundColor: "#f0f0f0"
		property string separatorColor: "#808080"
		property string fontName: "sans serif"
	}
}
