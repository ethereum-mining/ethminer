pragma Singleton
import QtQuick 2.0

QtObject {

	function absoluteSize(rel)
	{
		return systemPointSize + rel;
	}

	property QtObject general: QtObject {
		property int statusFontSize: absoluteSize(-1)
		property int logLinkFontSize: absoluteSize(-2)
	}
}
