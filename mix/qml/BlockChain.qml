import QtQuick 2.2
import QtQuick.Controls 1.1
import QtQuick.Controls.Styles 1.1
import QtQuick.Dialogs 1.1
import QtQuick.Layouts 1.1
import Qt.labs.settings 1.0
import org.ethereum.qml.QEther 1.0
import "js/Debugger.js" as Debugger
import "js/ErrorLocationFormater.js" as ErrorLocationFormater
import "js/TransactionHelper.js" as TransactionHelper
import "js/QEtherHelper.js" as QEtherHelper
import "."

ColumnLayout {
    id: blockChainPanel
    property variant model
    spacing: 0
    property int previousWidth
    onWidthChanged:
    {

        if (width <= 630 || previousWidth <= 630)
        {
            fromWidth = 100
            toWidth = 100
            valueWidth = 200
        }
        else
        {
            var diff = (width - previousWidth) / 3;
            fromWidth = fromWidth + diff < 100 ? 100 : fromWidth + diff
            toWidth = toWidth + diff < 100 ? 100 : toWidth + diff
            valueWidth = valueWidth + diff < 200 ? 200 : valueWidth + diff
        }
        previousWidth = width
    }

    function load(scenario)
    {
        if (!scenario)
            return;
        model = scenario
        blockModel.clear()
        for (var b in model.blocks)
            blockModel.append(model.blocks[b])
        previousWidth = width
    }

    property int statusWidth: 30
    property int fromWidth: 100
    property int toWidth: 100
    property int valueWidth: 200
    property int logsWidth: 50
    property int debugActionWidth: 50
    property int horizontalMargin: 10
    property int cellSpacing: 10

    RowLayout
    {
        id: header
        spacing: 0
        Layout.preferredHeight: 25
        Image {
            id: debugImage
            source: "qrc:/qml/img/recycle-icon@2x.png"
            Layout.preferredWidth: statusWidth
            Layout.preferredHeight: 25
            fillMode: Image.PreserveAspectFit
        }
        Rectangle
        {
            Layout.preferredWidth: fromWidth + cellSpacing
            Label
            {
                anchors.verticalCenter: parent.verticalCenter
                text: "From"
                anchors.left: parent.left
                anchors.leftMargin: horizontalMargin + 5
            }
        }
        Label
        {
            text: "To"
            Layout.preferredWidth: toWidth + cellSpacing
        }
        Label
        {
            text: "Value"
            Layout.preferredWidth: valueWidth + cellSpacing
        }
        Label
        {
            text: "Logs"
            Layout.preferredWidth: logsWidth + cellSpacing
        }
        Label
        {
            text: "Action"
            Layout.preferredWidth: debugActionWidth
        }
    }

    Rectangle
    {
        Layout.preferredHeight: 500
        Layout.preferredWidth: parent.width
        //height: 500
        border.color: "#cccccc"
        border.width: 2
        color: "white"
        ScrollView
        {
            id: blockChainScrollView
            anchors.fill: parent
            anchors.topMargin: 10
            ColumnLayout
            {
                id: blockChainLayout
                width: parent.width
                spacing: 10
                Repeater // List of blocks
                {
                    id: blockChainRepeater
                    model: blockModel
                    Block
                    {
                        Layout.preferredWidth: blockChainScrollView.width
                        Layout.preferredHeight:
                        {
                            return calculateHeight()
                        }

                        transactions:
                        {
                            if (index >= 0)
                                return blockModel.get(index).transactions
                            else
                                return []
                        }

                        status:
                        {
                            if (index >= 0)
                                return blockModel.get(index).status
                            else
                                return ""
                        }

                        number:
                        {
                            if (index >= 0)
                                return blockModel.get(index).number
                            else
                                return 0
                        }
                    }
                }
            }
        }
    }

    ListModel
    {
        id: blockModel

        function appendBlock(block)
        {
            blockModel.append(block);
        }

        function appendTransaction(tr)
        {
            blockModel.get(blockModel.count - 1).transactions.append(tr)
        }

        function removeTransaction(blockIndex, trIndex)
        {
            blockModel.get(blockIndex).transactions.remove(trIndex)
        }

        function removeLastBlock()
        {
            blockModel.remove(blockModel.count - 1)
        }

        function removeBlock(index)
        {
            blockModel.remove(index)
        }

        function getTransaction(block, tr)
        {
            return blockModel.get(block).transactions.get(tr)
        }

        function setTransaction(blockIndex, trIndex, tr)
        {
            blockModel.get(blockIndex).transactions.set(trIndex, tr)
        }

        function setTransactionProperty(blockIndex, trIndex, propertyName, value)
        {
            blockModel.get(blockIndex).transactions.set(trIndex, { propertyName: value })
        }
    }

    Rectangle
    {
        Layout.preferredWidth: parent.width
        RowLayout
        {
            width: parent.width
            anchors.top: parent.top
            anchors.topMargin: 10
            ScenarioButton {
                id: rebuild
                text: qsTr("Rebuild")
                onClicked:
                {
                    for (var j = 0; j < model.blocks.length; j++)
                    {
                        for (var k = 0; k < model.blocks[j].transactions.length; k++)
                        {
                            if (!blockModel.get(j).transactions.get(k).saveStatus)
                            {
                                model.blocks[j].transactions.splice(k, 1)
                                blockModel.removeTransaction(j, k)
                                if (model.blocks[j].transactions.length === 0)
                                {
                                    model.blocks.splice(j, 1);
                                    blockModel.removeBlock(j);
                                }
                            }
                        }
                    }
                    clientModel.setupScenario(model);
                }

                Layout.preferredWidth: 100
                Layout.preferredHeight: 30
                buttonShortcut: ""
                sourceImg: "qrc:/qml/img/recycle-icon@2x.png"
            }

            ScenarioButton {
                id: addTransaction
                text: qsTr("Add Transaction")
                onClicked:
                {
                    var lastBlock = model.blocks[model.blocks.length - 1];
                    if (lastBlock.status === "mined")
                        model.blocks.push(projectModel.stateListModel.createEmptyBlock());
                    var item = TransactionHelper.defaultTransaction()
                    transactionDialog.stateAccounts = model.accounts
                    transactionDialog.open(model.blocks[model.blocks.length - 1].transactions.length, model.blocks.length - 1, item)
                }
                Layout.preferredWidth: 100
                Layout.preferredHeight: 30
                buttonShortcut: ""
                sourceImg: "qrc:/qml/img/recycle-icon@2x.png"
            }

            ScenarioButton {
                id: addBlockBtn
                text: qsTr("Add Block")
                onClicked:
                {
                    var lastBlock = model.blocks[model.blocks.length - 1]
                    if (lastBlock.status === "pending")
                        clientModel.mine()
                    else
                        addNewBlock()
                }

                function addNewBlock()
                {
                    var block = projectModel.stateListModel.createEmptyBlock()
                    model.blocks.push(block)
                    blockModel.appendBlock(block)
                }
                Layout.preferredWidth: 100
                Layout.preferredHeight: 30
                buttonShortcut: ""
                sourceImg: "qrc:/qml/img/recycle-icon@2x.png"
            }

            Connections
            {
                target: clientModel
                onNewBlock:
                {
                    if (!clientModel.running)
                    {
                        var lastBlock = model.blocks[model.blocks.length - 1]
                        lastBlock.status = "mined"
                        lastBlock.number = model.blocks.length
                        var lastB = blockModel.get(model.blocks.length - 1)
                        lastB.status = "mined"
                        lastB.number = model.blocks.length
                        addBlockBtn.addNewBlock()
                    }
                }
                onStateCleared:
                {
                }
                onNewRecord:
                {
                    var blockIndex =  parseInt(_r.transactionIndex.split(":")[0]) - 1
                    var trIndex = parseInt(_r.transactionIndex.split(":")[1])
                    if (blockIndex <= model.blocks.length)
                    {
                        var item = model.blocks[blockIndex]
                        if (trIndex <= item.transactions.length)
                        {
                            var tr = item.transactions[trIndex]
                            tr.returned = _r.returned
                            tr.recordIndex = _r.recordIndex
                            tr.logs = _r.logs
                            var trModel = blockModel.getTransaction(blockIndex, trIndex)
                            trModel.returned = _r.returned
                            trModel.recordIndex = _r.recordIndex
                            trModel.logs = _r.logs
                            blockModel.setTransaction(blockIndex, trIndex, trModel)
                            return;
                        }
                    }

                    // tr is not in the list. coming from JavaScript
                    var itemTr = TransactionHelper.defaultTransaction()
                    itemTr.saveStatus = false
                    itemTr.functionId = _r.function
                    itemTr.contractId = _r.contract
                    itemTr.gasAuto = true
                    itemTr.parameters = _r.parameters
                    itemTr.isContractCreation = itemTr.functionId === itemTr.contractId
                    itemTr.label = _r.label
                    itemTr.isFunctionCall = itemTr.functionId !== ""
                    itemTr.returned = _r.returned
                    itemTr.value = QEtherHelper.createEther(_r.value, QEther.Wei)
                    itemTr.sender = _r.sender
                    itemTr.recordIndex = _r.recordIndex
                    itemTr.logs = _r.logs
                    model.blocks[model.blocks.length - 1].transactions.push(itemTr)
                    blockModel.appendTransaction(itemTr)
                }
                onMiningComplete:
                {
                }
            }

            ScenarioButton {
                id: newAccount
                text: qsTr("New Account")
                onClicked: {
                    model.accounts.push(projectModel.stateListModel.newAccount("1000000", QEther.Ether))
                }
                Layout.preferredWidth: 100
                Layout.preferredHeight: 30
                buttonShortcut: ""
                sourceImg: "qrc:/qml/img/recycle-icon@2x.png"
            }
        }
    }

    TransactionDialog {
        id: transactionDialog
        onAccepted: {
            var item = transactionDialog.getItem()
            var lastBlock = model.blocks[model.blocks.length - 1];
            if (lastBlock.status === "mined")
                model.blocks.push(projectModel.stateListModel.createEmptyBlock());
            model.blocks[model.blocks.length - 1].transactions.push(item)
            blockModel.appendTransaction(item)
            if (!clientModel.running)
                clientModel.executeTr(item)
        }
    }
}


