import QtQuick 2.2
import QtQuick.Controls 1.1
import QtQuick.Layouts 1.1
import QtQuick.Window 2.0
import QtQuick.Controls.Styles 1.3
import org.ethereum.qml.QEther 1.0
import "js/TransactionHelper.js" as TransactionHelper
import "js/ProjectModel.js" as ProjectModelCode
import "."

Window {
	id: modalDeploymentDialog
	modality: Qt.ApplicationModal
	width: 520
	height: 200
	visible: false
	property alias applicationUrlEth: applicationUrlEth.text
	property alias applicationUrlHttp: applicationUrlHttp.text
	color: Style.generic.layout.backgroundColor

	function close()
	{
		visible = false;
	}

	function open()
	{
		modalDeploymentDialog.setX((Screen.width - width) / 2);
		modalDeploymentDialog.setY((Screen.height - height) / 2);
		visible = true;
	}

	GridLayout
	{
		columns: 2
		anchors.top: parent.top
		anchors.left: parent.left
		anchors.topMargin: 10
		anchors.leftMargin: 10
		anchors.rightMargin: 10
		DefaultLabel
		{
			text: qsTr("Eth URL: ")
		}

		DefaultTextField
		{
			id: applicationUrlEth
		}

		DefaultLabel
		{
			text: qsTr("Http URL: ")
		}

		DefaultTextField
		{
			id: applicationUrlHttp
		}
	}

	RowLayout
	{
		anchors.bottom: parent.bottom
		anchors.right: parent.right;
		anchors.bottomMargin: 10
		Button {
			text: qsTr("Deploy");
			enabled: Object.keys(projectModel.deploymentAddresses).length === 0
			onClicked: {
				ProjectModelCode.startDeployProject();
			}
		}

		Button {
			text: qsTr("Rebuild Package");
			enabled: Object.keys(projectModel.deploymentAddresses).length > 0 && applicationUrlHttp.text !== ""
			onClicked: {
				var date = new Date();
				var deploymentId = date.toLocaleString(Qt.locale(), "ddMMyyHHmmsszzz");
				ProjectModelCode.finalizeDeployment(deploymentId, projectModel.deploymentAddresses);
			}
		}

		Button {
			text: qsTr("Close");
			onClicked: close();
		}
	}
}
