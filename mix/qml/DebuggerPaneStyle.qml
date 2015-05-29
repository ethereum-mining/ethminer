import QtQuick 2.0

QtObject {

	function absoluteSize(rel)
	{
		return systemPointSize + rel;
	}

	property QtObject general: QtObject {
		property int basicFontSize: absoluteSize(-2)
		property string basicColor: "#4a4a4a"
		property string basicFont: "monospace"
		property int dataDumpFontSize: absoluteSize(-3)
	}
}
