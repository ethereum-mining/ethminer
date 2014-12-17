import QtQuick 2.2
import QtQuick.Window 2.0

Window
{
	id: modalDialog
	title: ""
	modality: Qt.WindowModal
	height: 400
	width: 700
	visible: false
	Loader
	{
		focus: true
		id: modalDialogContent
		objectName: "modalDialogContent"
		anchors.fill: parent
	}
	function open()
	{
		visible = true;
	}
	function close()
	{
		visible = false;
		modalDialogContent.source = "";
		modalDialogContent.sourceComponent = undefined;
		modalDialog.destroy();
	}
}
