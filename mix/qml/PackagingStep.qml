import QtQuick 2.0
import QtQuick.Layouts 1.0
import QtQuick.Controls 1.1
import QtQuick.Controls.Styles 1.3
import QtQuick.Dialogs 1.1
import Qt.labs.settings 1.0
import "js/TransactionHelper.js" as TransactionHelper
import "js/NetworkDeployment.js" as NetworkDeploymentCode
import "js/QEtherHelper.js" as QEtherHelper

Rectangle {
	property variant paramsModel: []
	property variant worker
	color: "#E3E3E3E3"
	anchors.fill: parent

	property string packageHash
	property string packageBase64
	property alias localPackageUrl: localPackageUrl.text
	property string deploymentId
	property string packageDir

	Settings {
		property alias localUrl: localPackageUrl.text
	}


	function show()
	{
		visible = true
	}

	FileDialog {
		id: ressourcesFolder
		visible: false
		title: qsTr("Please choose a path")
		selectFolder: true
		property variant target
		onAccepted: {
			var u = ressourcesFolder.fileUrl.toString();
			if (u.indexOf("file://") == 0)
				u = u.substring(7, u.length)
			if (Qt.platform.os == "windows" && u.indexOf("/") == 0)
				u = u.substring(1, u.length);
			target.text = u;
		}
	}

	ColumnLayout
	{
		anchors.top: parent.top
		anchors.topMargin: 10
		width: parent.width

		id: col
		spacing: 20

		Label
		{
			anchors.top: parent.top
			Layout.fillWidth: true
			anchors.left: parent.left
			anchors.leftMargin: 10
			text: qsTr("Upload and update your Dapp assets")
		}

		RowLayout
		{
			Layout.fillWidth: true
			Layout.preferredHeight: 20
			Rectangle
			{
				Layout.preferredWidth: col.width / 2
				Label
				{
					text: qsTr("Save Package to")
					anchors.right: parent.right
					anchors.verticalCenter: parent.verticalCenter
				}
			}

			DefaultTextField
			{
				id: packageFolder
				visible: true
				Layout.preferredWidth: 150
				text: projectPath + "package/"
			}

			Button
			{
				text: qsTr("select")
				onClicked: {
					ressourcesFolder.target = packageFolder
					ressourcesFolder.open()
				}
			}
		}

		Rectangle
		{
			Layout.fillWidth: true
			Layout.preferredHeight: 20
			color: "transparent"
			Button
			{
				Layout.preferredWidth: 200
				anchors.horizontalCenter: parent.horizontalCenter
				text: qsTr("Generate Package")
				onClicked:
				{
					NetworkDeploymentCode.packageDapp(projectModel.deploymentAddresses)
				}
			}
		}

		RowLayout
		{
			Layout.fillWidth: true
			Layout.preferredHeight: 20
			Rectangle
			{
				Layout.preferredWidth: col.width / 2
				Label
				{
					text: qsTr("Local package URL")
					anchors.right: parent.right
					anchors.verticalCenter: parent.verticalCenter
				}
			}

			DefaultTextField
			{
				id: localPackageUrl
				Layout.preferredWidth: 235
				readOnly: true
			}
		}

		Label
		{
			Layout.preferredWidth: 300
			anchors.horizontalCenter: parent.horizontalCenter
			text: qsTr("You have to upload the package to a remote folder, or use a service like pastebin")
			wrapMode: Text.WordWrap
			clip: true
		}

		Rectangle
		{
			color: "transparent"
			Layout.fillWidth: true
			Layout.preferredHeight: 20
			Button
			{
				Layout.preferredWidth: 200
				anchors.horizontalCenter: parent.horizontalCenter
				text: qsTr("Copy Base64")
				onClicked:
				{
					clipboard.text = deploymentDialog.packageStep.packageBase64;
				}
			}
		}

		Rectangle
		{
			color: "transparent"
			Layout.fillWidth: true
			Layout.preferredHeight: 20
			Button
			{
				Layout.preferredWidth: 200
				anchors.horizontalCenter: parent.horizontalCenter
				text: qsTr("Open pastebin")
				onClicked:
				{
					Qt.openUrlExternally("http://pastebin.com/");
				}
			}
		}
	}
}









