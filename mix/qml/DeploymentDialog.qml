import QtQuick 2.2
import QtQuick.Controls 1.1
import QtQuick.Layouts 1.1
import QtQuick.Window 2.0
import QtQuick.Controls.Styles 1.3
import org.ethereum.qml.QEther 1.0
import "js/TransactionHelper.js" as TransactionHelper
import "."

Window {
	id: modalTransactionDialog
	modality: Qt.ApplicationModal
	width: 520
	height: 300
	visible: false
	property alias applicationUrlEth: applicationUrlEth.text
	property alias applicationUrlHttp: applicationUrlHttp.text

	signal accepted

	function close()
	{
		visible = false;
	}

	function open()
	{
		visible = true;
	}

	ColumnLayout
	{
		anchors.fill: parent
		RowLayout
		{
			height: 40
			DefaultLabel
			{
				text: qsTr("Fill in eth application URL")
			}

			Rectangle
			{
				TextField
				{
					id: applicationUrlEth
				}
			}
		}

		RowLayout
		{
			height: 40
			DefaultLabel
			{
				text: qsTr("Fill in http application URL")
			}

			Rectangle
			{
				TextField
				{
					id: applicationUrlHttp
				}
			}
		}

		RowLayout
		{
			anchors.bottom: parent.bottom
			anchors.right: parent.right;

			Button {
				text: qsTr("OK");
				onClicked: {
					close();
					accepted();
				}
			}
			Button {
				text: qsTr("Cancel");
				onClicked: close();
			}
		}
	}


}
