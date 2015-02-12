pragma Singleton
import QtQuick 2.0

QtObject {

	function absoluteSize(rel)
	{
		return systemPointSize + rel;
	}

	property QtObject generic: QtObject {
		property QtObject layout: QtObject {
			property string separatorColor: "#808080"
		}
		property QtObject size: QtObject {
			property string titlePointSize: absoluteSize(0)
		}
	}
}
