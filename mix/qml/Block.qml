import QtQuick 2.2
import QtQuick.Controls 1.1
import QtQuick.Controls.Styles 1.1
import QtQuick.Dialogs 1.1
import QtQuick.Layouts 1.1
import Qt.labs.settings 1.0
import "js/Debugger.js" as Debugger
import "js/ErrorLocationFormater.js" as ErrorLocationFormater
import "."

ColumnLayout
{
    property variant transactions
    property string status
    property int number
    Rectangle
    {
        width: parent.width
        height: 50
        anchors.left: parent.left
        anchors.leftMargin: statusWidth
        Label {
            text:
            {
                if (status === "mined")
                    return qsTr("BLOCK") + " " + number
                else
                    return qsTr("BLOCK") + " pending"
            }

            anchors.left: parent.left
        }
    }

    Repeater // List of transactions
    {
        id: transactionRepeater
        model: transactions
        Row
        {
            height: 50
            Rectangle
            {
                id: trSaveStatus
                color: "transparent"
                CheckBox
                {
                    id: saveStatus
                    checked: {
                        if (index >= 0)
                            return transactions.get(index).saveStatus
                        else
                            return true
                    }
                    onCheckedChanged:
                    {
                        if (index >= 0)
                            transactions.get(index).saveStatus = checked
                    }
                }
            }

            Rectangle
            {
                width: parent.width
                height: 50
                color: "#cccccc"
                radius: 4
                Row
                {
                    Label
                    {
                        id: status
                        width: statusWidth
                    }
                    Label
                    {
                        id: hash
                        width: fromWidth
                        text: {
                            if (index >= 0)
                                return transactions.get(index).sender
                            else
                                return ""
                        }

                        clip: true
                    }
                    Label
                    {
                        id: func
                        text: {
                            if (index >= 0)
                                parent.userFrienldyToken(transactions.get(index).label)
                            else
                                return ""
                        }

                        width: toWidth
                        clip: true
                    }

                    function userFrienldyToken(value)
                    {
                        if (value && value.indexOf("<") === 0)
                            return value.split(" - ")[0].replace("<", "") + "." + value.split("> ")[1] + "()";
                        else
                            return value
                    }

                    Label
                    {
                        id: returnValue
                        width: valueWidth
                        text: {
                            if (index >= 0 && transactions.get(index).returned)
                                return transactions.get(index).returned
                            else
                                return ""
                        }
                        clip: true
                    }

                    Button
                    {
                        id: debug
                        width: logsWidth
                        text: "debug"
                        onClicked:
                        {
                            clientModel.debugRecord(transactions.get(index).recordIndex);
                        }
                    }

                    Label
                    {
                        id: logs
                        width: logsWidth
                    }
                }
            }
        }
    }
}

