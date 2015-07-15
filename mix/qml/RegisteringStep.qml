import QtQuick 2.0
import QtQuick.Layouts 1.0
import QtQuick.Controls 1.1
import QtQuick.Controls.Styles 1.3
import org.ethereum.qml.QEther 1.0
import Qt.labs.settings 1.0
import "js/TransactionHelper.js" as TransactionHelper
import "js/NetworkDeployment.js" as NetworkDeploymentCode
import "js/QEtherHelper.js" as QEtherHelper
import "."

Rectangle {
	property variant worker
	property string eth: registrarAddr.text
	property int ownedRegistrarDeployGas: 1179075 // TODO: Use sol library to calculate gas requirement for each tr.
	property int ownedRegistrarSetSubRegistrarGas: 50000
	property int ownedRegistrarSetContentHashGas: 50000
	property int urlHintSuggestUrlGas: 70000

	color: "#E3E3E3E3"
	anchors.fill: parent

	function show()
	{
		ctrRegisterLabel.calculateRegisterGas()
		applicationUrlEthCtrl.text = projectModel.applicationUrlEth
		applicationUrlHttpCtrl.text = projectModel.applicationUrlHttp
		visible = true
	}

	ColumnLayout
	{
		anchors.top: parent.top
		width: parent.width
		anchors.topMargin: 10
		id: col
		spacing: 20
		Label
		{
			anchors.top: parent.top
			anchors.left: parent.left
			anchors.leftMargin: 10
			Layout.fillWidth: true
			text: qsTr("Register your Dapp on the Name registrar Contract")
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
					text: qsTr("Root Registrar address")
					anchors.right: parent.right
					anchors.verticalCenter: parent.verticalCenter
				}
			}

			DefaultTextField
			{
				id: registrarAddr
				text: "be5aac309199b4e0740ff52d4b28dc948cf44bea" //"c6d9d2cd449a754c494264e1809c50e34d64562b"
				visible: true
				Layout.preferredWidth: 235
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
					text: qsTr("Htpp URL")
					anchors.right: parent.right
					anchors.verticalCenter: parent.verticalCenter
				}
			}

			DefaultTextField
			{
				id: applicationUrlHttpCtrl
				Layout.preferredWidth: 235
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
					text: qsTr("Gas to use for dapp registration")
					anchors.right: parent.right
					anchors.verticalCenter: parent.verticalCenter
					id: ctrRegisterLabel
					function calculateRegisterGas()
					{
						if (!modalDeploymentDialog.visible)
							return;
						appUrlFormatted.text = NetworkDeploymentCode.formatAppUrl(applicationUrlEthCtrl.text).join('/');
						NetworkDeploymentCode.checkPathCreationCost(applicationUrlEthCtrl.text, function(pathCreationCost)
						{
							var ether = QEtherHelper.createBigInt(pathCreationCost);
							var gasTotal = ether.multiply(worker.gasPriceInt);
							gasToUseDeployInput.value = QEtherHelper.createEther(gasTotal.value(), QEther.Wei, parent);
							gasToUseDeployInput.update();
						});
					}
				}
			}

			Ether
			{
				id: gasToUseDeployInput
				displayUnitSelection: true
				displayFormattedValue: true
				edit: true
				Layout.preferredWidth: 235
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
					text: qsTr("Ethereum URL")
					anchors.right: parent.right
					anchors.verticalCenter: parent.verticalCenter
				}
			}

			Rectangle
			{
				height: 25
				color: "transparent"
				Layout.preferredWidth: 235
				DefaultTextField
				{
					width: 235
					id: applicationUrlEthCtrl
					onTextChanged: {
						ctrRegisterLabel.calculateRegisterGas();
					}
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
					text: qsTr("Formatted Ethereum URL")
					anchors.right: parent.right
					anchors.verticalCenter: parent.verticalCenter
				}
			}

			DefaultLabel
			{
				id: appUrlFormatted
				anchors.verticalCenter: parent.verticalCenter;
				anchors.top: applicationUrlEthCtrl.bottom
				anchors.topMargin: 10
				font.italic: true
				font.pointSize: appStyle.absoluteSize(-1)
			}
		}
	}

	RowLayout
	{
		anchors.bottom: parent.bottom
		width: parent.width
		anchors.bottomMargin: 8
		Button
		{
			anchors.right: parent.right
			anchors.rightMargin: 10
			text: qsTr("Register Dapp")
			width: 30
			onClicked:
			{
				var inError = [];
				var ethUrl = NetworkDeploymentCode.formatAppUrl(applicationUrlEthCtrl.text);
				for (var k in ethUrl)
				{
					if (ethUrl[k].length > 32)
						inError.push(qsTr("Member too long: " + ethUrl[k]) + "\n");
				}
				if (!worker.stopForInputError(inError))
				{
					NetworkDeploymentCode.registerDapp(ethUrl, function(){
						projectModel.applicationUrlEth = applicationUrlEthCtrl.text
						projectModel.saveProject()
					})
				}

				if (applicationUrlHttp.text === "" || deploymentDialog.packageHash === "")
				{
					deployDialog.title = text;
					deployDialog.text = qsTr("Please provide the link where the resources are stored and ensure the package is aleary built using the deployment step.")
					deployDialog.open();
					return;
				}
				inError = [];
				if (applicationUrlHttpCtrl.text.length > 32)
					inError.push(qsTr(applicationUrlHttpCtrl.text));
				if (!worker.stopForInputError(inError))
				{
					registerToUrlHint(applicationUrlHttpCtrl.text, function(){
						projectModel.applicationUrlHttp = applicationUrlHttpCtrl.text
						projectModel.saveProject()
					})
				}
			}
		}
	}
}

