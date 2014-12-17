import QtQuick 2.2
import QtQuick.Window 2.0

Window
{
	id: alertMessageDialog
	title: ""
	modality: Qt.WindowModal
	height: 150
	width: 200
	visible: false
	Loader
	{
		focus: true
		id: alertMessageDialogContent
		objectName: "alertMessageDialogContent"
		anchors.fill: parent
	}
	function open()
	{
		visible = true
	}
	function close()
	{
		visible = false;
		alertMessageDialogContent.source = "";
		alertMessageDialogContent.sourceComponent = undefined;
		alertMessageDialog.destroy();
	}
}
