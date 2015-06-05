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

Column {
    id: blockChainPanel
    property variant model
    spacing: 5

    function load(scenario)
    {

        if (!scenario)
            return;
        model = scenario
        blockModel.clear()
        for (var b in model.blocks)
            blockModel.append(model.blocks[b])
    }

    property int statusWidth: 50
    property int fromWidth: 100
    property int toWidth: 250
    property int valueWidth: 100
    property int logsWidth: 50

    Row
    {
        id: header
        width: parent.width
        Label
        {
            text: "Status"
            width: statusWidth
        }
        Label
        {
            text: "From"
            width: fromWidth
        }
        Label
        {
            text: "To"
            width: toWidth
        }
        Label
        {
            text: "Value"
            width: valueWidth
        }
        Label
        {
            text: "Logs"
            width: logsWidth
        }
    }

    Rectangle
    {
        width: parent.width
        height: 500
        border.color: "#cccccc"
        border.width: 2
        color: "white"
        ScrollView
        {
            width: parent.width
            height: parent.height
            ColumnLayout
            {
                Repeater // List of blocks
                {
                    id: blockChainRepeater
                    width: parent.width
                    model: blockModel
                    Block
                    {
                        height:
                        {
                            if (index >= 0)
                                return 50 + 50 * blockModel.get(index).transactions.length
                            else
                                return 0
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
            console.log(blockIndex)
            console.log(trIndex)
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
            return blockModel.get(block - 1).transactions.get(tr)
        }
    }

    RowLayout
    {
        width: parent.width
        Button {
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
                                model.blocks[j].splice(j, 1);
                                blockModel.removeBlock(j);
                            }
                        }
                    }
                }
                clientModel.setupScenario(model);
            }
        }

        Button {
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
        }

        Button {
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
                var blockIndex = _r.transactionIndex.split(":")[0]
                var trIndex = _r.transactionIndex.split(":")[1]
                if (parseInt(blockIndex) <= model.blocks.length)
                {
                    var item = model.blocks[parseInt(blockIndex) - 1];
                    if (parseInt(trIndex) <= item.transactions.length)
                    {
                        var tr = item.transactions[parseInt(trIndex)];
                        tr.returned = _r.returned;
                        blockModel.getTransaction(blockIndex, trIndex).returned = _r.returned;
                        tr.recordIndex = _r.recordIndex;
                        blockModel.getTransaction(blockIndex, trIndex).recordIndex = _r.recordIndex;
                        return;
                    }
                }

                // tr is not in the list. coming from JavaScript
                var itemTr = TransactionHelper.defaultTransaction()
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

                model.blocks[model.blocks.length - 1].transactions.push(itemTr)
                blockModel.appendTransaction(itemTr)
            }
            onMiningComplete:
            {
            }
        }

        Button {
            id: newAccount
            text: qsTr("New Account")
            onClicked: {
                model.accounts.push(projectModel.stateListModel.newAccount("1000000", QEther.Ether))
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


