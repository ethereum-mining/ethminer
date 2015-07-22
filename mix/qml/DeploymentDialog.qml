import QtQuick 2.2
import QtQuick.Controls 1.1
import QtQuick.Layouts 1.1
import QtQuick.Window 2.0
import QtQuick.Dialogs 1.2
import QtQuick.Controls.Styles 1.3
import org.ethereum.qml.QEther 1.0
import "js/TransactionHelper.js" as TransactionHelper
import "js/NetworkDeployment.js" as NetworkDeploymentCode
import "js/QEtherHelper.js" as QEtherHelper
import "."


Dialog {
	id: modalDeploymentDialog
	modality: Qt.ApplicationModal
	width: 1000
	height: 450
	visible: false

	property alias deployStep: deployStep
	property alias packageStep: packageStep
	property alias registerStep: registerStep
	property alias worker: worker
	property alias steps: steps

	function close()
	{
		visible = false;
	}

	function open()
	{
		deployStep.visible = false
		packageStep.visible = false
		registerStep.visible = false
		steps.init()
		worker.renewCtx()
		visible = true;
	}

	DeploymentWorker
	{
		id: worker
	}

	contentItem: Rectangle {
		color: appStyle.generic.layout.backgroundColor
		anchors.fill: parent
		ColumnLayout
		{
			spacing: 5
			anchors.fill: parent
			anchors.margins: 10

			RowLayout
			{
				id: explanation
				Layout.preferredWidth: parent.width - 50
				Layout.preferredHeight: 50
				Label
				{
					anchors.centerIn: parent
					text: qsTr("Putting your dapp live is a multi step process. You can read more about it on the 'guide to uploading'.")
				}
			}

			RowLayout
			{
				ColumnLayout
				{
					Layout.preferredHeight: parent.height - 50
					Layout.preferredWidth: 200
					DeploymentDialogSteps
					{
						id: steps
						worker: worker
					}
				}

				Connections
				{
					target: steps
					property variant selected
					onSelected:
					{
						if (selected)
							selected.visible = false
						switch (step)
						{
						case "deploy":
						{
							selected = deployStep
							break;
						}
						case "package":
						{
							selected = packageStep
							break;
						}
						case "register":
						{
							selected = registerStep
							break;
						}
						}
						selected.show()
					}
				}

				ColumnLayout
				{
					Layout.preferredHeight: parent.height - 50
					Layout.preferredWidth: parent.width - 200
					DeployContractStep
					{
						id: deployStep
						visible: false
						worker: worker
					}

					PackagingStep
					{
						id: packageStep
						visible: false
						worker: worker
					}

					RegisteringStep
					{
						id: registerStep
						visible: false
						worker: worker
					}
				}
			}

			Rectangle
			{
				Layout.fillWidth: true
				Layout.preferredHeight: 30
				color: "transparent"
				Button
				{
					text: qsTr("Cancel")
					anchors.right: parent.right
					anchors.rightMargin: 10
					onClicked:
					{
						modalDeploymentDialog.close()
					}
				}
			}

		}
	}

	MessageDialog {
		id: deployDialog
		standardButtons: StandardButton.Ok
		icon: StandardIcon.Warning
	}

	MessageDialog {
		id: errorDialog
		standardButtons: StandardButton.Ok
		icon: StandardIcon.Critical
	}
}
