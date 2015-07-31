import QtQuick 2.0
import QtQuick.Layouts 1.0
import QtQuick.Controls 1.1
import QtQuick.Controls.Styles 1.3
import Qt.labs.settings 1.0
import "js/TransactionHelper.js" as TransactionHelper
import "js/NetworkDeployment.js" as NetworkDeploymentCode
import "js/QEtherHelper.js" as QEtherHelper
import org.ethereum.qml.QEther 1.0

Rectangle {
	property variant paramsModel: []
	property variant worker
	property variant gas: []
	property alias gasPrice: gasPriceInput
	color: "#E3E3E3E3"
	signal deployed
	anchors.fill: parent
	id: root

	property int labelWidth: 150
	property bool verifyDeploy: true

	function show()
	{
		visible = true
		contractList.currentIndex = 0
		contractList.change()
		accountsModel.clear()
		for (var k in worker.accounts)
			accountsModel.append(worker.accounts[k])

		if (worker.currentAccount === "" && worker.accounts.length > 0)
		{
			worker.currentAccount = worker.accounts[0].id
			accountsList.currentIndex = 0
		}

		verifyDeployedContract()
		deployedAddresses.refresh()

		worker.renewCtx()
		verifyDeploy = true
		worker.pooler.onTriggered.connect(function() {
			if (root.visible && verifyDeploy)
				verifyDeployedContract();
		})
	}

	function verifyDeployedContract()
	{
		if (projectModel.deployBlockNumber !== -1)
		{
			worker.verifyHashes(projectModel.deploymentTrHashes, function (bn, trLost)
			{
				root.updateVerification(bn, trLost)
			});
		}
	}

	function updateVerification(blockNumber, trLost)
	{
		var nb = parseInt(blockNumber - projectModel.deployBlockNumber)
		verificationTextArea.visible = false
		verificationLabel.visible = true
		if (nb >= 10)
		{
			verificationLabel.text = qsTr("contracts deployment verified")
			verificationLabel.color = "green"
		}
		else
		{
			verificationLabel.text = nb
			if (trLost.length > 0)
			{
				verifyDeploy = false
				verificationTextArea.visible = true
				verificationLabel.visible = false
				verificationTextArea.text = ""
				deploymentStepChanged("following transactions are invalidated:")
				verificationTextArea.text += "\n" + qsTr("Transactions lost") + "\n"
				verificationTextArea.textColor = "red"
				for (var k in trLost)
				{
					deploymentStepChanged(trLost[k])
					verificationTextArea.text += trLost[k] + "\n"
				}
			}
		}
	}

	RowLayout
	{
		anchors.fill: parent
		anchors.margins: 10
		ColumnLayout
		{
			anchors.top: parent.top
			Layout.preferredWidth: parent.width * 0.40 - 20
			Layout.fillHeight: true
			id: scenarioList

			Label
			{
				Layout.fillWidth: true
				text: qsTr("Pick Scenario to deploy")
			}

			ComboBox
			{
				id: contractList
				Layout.preferredWidth: parent.width - 20
				model: projectModel.stateListModel
				textRole: "title"
				onCurrentIndexChanged:
				{
					if (root.visible)
						change()
				}

				function change()
				{
					trListModel.clear()
					if (currentIndex > -1)
					{
						for (var k = 0; k < projectModel.stateListModel.get(currentIndex).blocks.count; k++)
						{
							for (var j = 0; j < projectModel.stateListModel.get(currentIndex).blocks.get(k).transactions.count; j++)
								trListModel.append(projectModel.stateListModel.get(currentIndex).blocks.get(k).transactions.get(j));
						}
						for (var k = 0; k < trListModel.count; k++)
							trList.itemAt(k).init()
						ctrDeployCtrLabel.calculateContractDeployGas();
					}
				}
			}

			Rectangle
			{
				Layout.fillHeight: true
				Layout.preferredWidth: parent.width - 20
				id: trContainer
				color: "white"
				border.color: "#cccccc"
				border.width: 1
				ScrollView
				{
					anchors.fill: parent
					horizontalScrollBarPolicy: Qt.ScrollBarAlwaysOff
					ColumnLayout
					{
						spacing: 0

						ListModel
						{
							id: trListModel
						}

						Repeater
						{
							id: trList
							model: trListModel
							ColumnLayout
							{
								Layout.fillWidth: true
								spacing: 5
								Layout.preferredHeight:
								{
									if (index > -1)
										return 20 + trListModel.get(index)["parameters"].count * 20
									else
										return 20
								}

								function init()
								{
									paramList.clear()
									if (trListModel.get(index).parameters)
									{
										for (var k in trListModel.get(index).parameters)
											paramList.append({ "name": k, "value": trListModel.get(index).parameters[k] })
									}
								}

								Label
								{
									id: trLabel
									Layout.preferredHeight: 20
									anchors.left: parent.left
									anchors.top: parent.top
									anchors.topMargin: 5
									anchors.leftMargin: 10
									text:
									{
										if (index > -1)
											return trListModel.get(index).label
										else
											return ""
									}
								}

								ListModel
								{
									id: paramList
								}

								Repeater
								{
									Layout.preferredHeight:
									{
										if (index > -1)
											return trListModel.get(index)["parameters"].count * 20
										else
											return 0
									}
									model: paramList
									Label
									{
										Layout.preferredHeight: 20
										anchors.left: parent.left
										anchors.leftMargin: 20
										text: name + "=" + value
										font.italic: true
									}
								}

								Rectangle
								{
									Layout.preferredWidth: scenarioList.width
									Layout.preferredHeight: 1
									color: "#cccccc"
								}
							}
						}
					}
				}
			}
		}

		ColumnLayout
		{
			anchors.top: parent.top
			Layout.preferredHeight: parent.height - 25
			ColumnLayout
			{
				anchors.top: parent.top
				Layout.preferredWidth: parent.width * 0.60
				Layout.fillHeight: true
				id: deploymentOption
				spacing: 8

				Label
				{
					anchors.left: parent.left
					anchors.leftMargin: 105
					text: qsTr("Deployment options")
				}

				ListModel
				{
					id: accountsModel
				}

				RowLayout
				{
					Layout.fillWidth: true
					Rectangle
					{
						width: labelWidth
						Label
						{
							text: qsTr("Account")
							anchors.right: parent.right
							anchors.verticalCenter: parent.verticalCenter
						}
					}

					ComboBox
					{
						id: accountsList
						textRole: "id"
						model: accountsModel
						Layout.preferredWidth: 235
						onCurrentTextChanged:
						{
							worker.currentAccount = currentText
							accountBalance.text = worker.balance(currentText).format()
						}
					}

					Label
					{
						id: accountBalance
					}
				}

				RowLayout
				{
					Layout.fillWidth: true
					Rectangle
					{
						width: labelWidth
						Label
						{
							text: qsTr("Gas Price")
							anchors.right: parent.right
							anchors.verticalCenter: parent.verticalCenter
						}
					}

					Ether
					{
						id: gasPriceInput
						displayUnitSelection: true
						displayFormattedValue: true
						edit: true

						function toHexWei()
						{
							return "0x" + gasPriceInput.value.toWei().hexValue()
						}
					}

					Connections
					{
						target: gasPriceInput
						onValueChanged:
						{
							ctrDeployCtrLabel.calculateContractDeployGas()
						}
						onAmountChanged:
						{
							ctrDeployCtrLabel.setCost()
						}
						onUnitChanged:
						{
							ctrDeployCtrLabel.setCost()
						}
					}

					Connections
					{
						target: worker
						id: gasPriceLoad
						property bool loaded: false
						onGasPriceLoaded:
						{
							gasPriceInput.value = QEtherHelper.createEther(worker.gasPriceInt.value(), QEther.Wei)
							gasPriceLoad.loaded = true
							ctrDeployCtrLabel.calculateContractDeployGas()
						}
					}
				}

				RowLayout
				{
					id: ctrDeployCtrLabel
					Layout.fillWidth: true
					property int cost
					function calculateContractDeployGas()
					{
						if (!root.visible)
							return;
						var sce = projectModel.stateListModel.getState(contractList.currentIndex)
						worker.estimateGas(sce, function(gas) {
							if (gasPriceLoad.loaded)
							{
								root.gas = gas
								cost = 0
								for (var k in gas)
									cost += gas[k]
								setCost()
							}
						});
					}

					function setCost()
					{
						var ether = QEtherHelper.createBigInt(cost);
						var gasTotal = ether.multiply(gasPriceInput.value);
						gasToUseInput.value = QEtherHelper.createEther(gasTotal.value(), QEther.Wei, parent);
					}

					Rectangle
					{
						width: labelWidth
						Label
						{
							text: qsTr("Deployment Cost")
							anchors.right: parent.right
							anchors.verticalCenter: parent.verticalCenter
						}
					}

					Ether
					{
						id: gasToUseInput
						displayUnitSelection: false
						displayFormattedValue: true
						edit: false
						Layout.preferredWidth: 350
					}
				}

				Rectangle
				{
					border.color: "#cccccc"
					border.width: 2
					Layout.fillWidth: true
					Layout.preferredHeight: parent.height + 25
					color: "transparent"
					id: rectDeploymentVariable
					ScrollView
					{
						anchors.fill: parent
                        anchors.topMargin: 4
                        anchors.bottomMargin: 4
						ColumnLayout
						{
							RowLayout
							{
								id: deployedRow
								Layout.fillWidth: true
								Rectangle
								{
									width: labelWidth
									Label
									{
										id: labelAddresses
										text: qsTr("Deployed Contracts")
										anchors.right: parent.right
										anchors.verticalCenter: parent.verticalCenter
									}
								}

								ColumnLayout
								{
									anchors.top: parent.top
									anchors.topMargin: 1
									width: parent.width
									id: deployedAddresses
									function refresh()
									{
										textAddresses.text = ""
										deployedRow.visible = Object.keys(projectModel.deploymentAddresses).length > 0
										textAddresses.text = JSON.stringify(projectModel.deploymentAddresses, null, ' ')
									}
									TextArea
									{
										anchors.fill: parent
										id: textAddresses
									}
								}
							}

							RowLayout
							{
								id: verificationRow
								Layout.fillWidth: true
								visible: Object.keys(projectModel.deploymentAddresses).length > 0
								Rectangle
								{
									width: labelWidth
									Label
									{
										text: qsTr("Verifications")
										anchors.right: parent.right
										anchors.verticalCenter: parent.verticalCenter
									}
								}

								TextArea
								{
									id: verificationTextArea
									visible: false
								}

								Label
								{
									id: verificationLabel
									visible: true
								}
							}
						}
					}
				}
			}

			Rectangle
			{
				Layout.preferredWidth: parent.width
				Layout.alignment: Qt.BottomEdge
				Button
				{
					Layout.preferredHeight: 22
					anchors.right: deployBtn.left
					text: qsTr("Reset")
					action: clearDeployAction
				}

				Action {
					id: clearDeployAction
					onTriggered: {
						worker.forceStopPooling()
						if (projectModel.deploymentDir && projectModel.deploymentDir !== "")
							fileIo.deleteDir(projectModel.deploymentDir)
						projectModel.cleanDeploymentStatus()
						deploymentDialog.steps.reset()
					}
				}

				Button
				{
					id: deployBtn
					anchors.right: parent.right
					text: qsTr("Deploy Contracts")
					onClicked:
					{
						projectModel.deployedScenarioIndex = contractList.currentIndex
						NetworkDeploymentCode.deployContracts(root.gas, gasPriceInput.toHexWei(), function(addresses, trHashes)
						{
							projectModel.deploymentTrHashes = trHashes
							worker.verifyHashes(trHashes, function (nb, trLost)
							{
								projectModel.deployBlockNumber = nb
								projectModel.saveProject()
								root.updateVerification(nb, trLost)
								root.deployed()
							})
							projectModel.deploymentAddresses = addresses
							projectModel.saveProject()
							deployedAddresses.refresh()
						});
					}
				}
			}
		}
	}
}

