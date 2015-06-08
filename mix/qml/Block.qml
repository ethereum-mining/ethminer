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
    id: root
    property variant transactions
    property string status
    property int number
    property int blockWidth: Layout.preferredWidth - statusWidth - horizontalMargin
    property int horizontalMargin: 10
    property int trHeight: 30
    spacing: 0

    RowLayout
    {
        Layout.preferredHeight: trHeight
        Layout.preferredWidth: blockWidth
        id: rowHeader
        Rectangle
        {
            color: "#DEDCDC"
            Layout.preferredWidth: blockWidth
            Layout.preferredHeight: trHeight
            radius: 4
            anchors.left: parent.left
            anchors.leftMargin: statusWidth + 5
            Label {
                anchors.verticalCenter: parent.verticalCenter
                anchors.left: parent.left
                anchors.leftMargin: horizontalMargin
                text:
                {
                    if (status === "mined")
                        return qsTr("BLOCK") + " " + number
                    else
                        return qsTr("BLOCK") + " pending"
                }
            }
        }
    }

    Repeater // List of transactions
    {
        id: transactionRepeater
        model: transactions

        RowLayout
        {
            Layout.preferredHeight: trHeight
            Rectangle
            {
                id: trSaveStatus
                Layout.preferredWidth: statusWidth
                Layout.preferredHeight: trHeight
                color: "transparent"
                property bool saveStatus

                Image {
                    id: saveStatusImage
                    source: "qrc:/qml/img/recycle-discard@2x.png"
                    width: statusWidth
                    fillMode: Image.PreserveAspectFit
                    anchors.verticalCenter: parent.verticalCenter
                    anchors.horizontalCenter: parent.horizontalCenter
                }

                Component.onCompleted:
                {
                    if (index >= 0)
                        saveStatus = transactions.get(index).saveStatus
                }

                onSaveStatusChanged:
                {
                    if (saveStatus)
                        saveStatusImage.source = "qrc:/qml/img/recycle-keep@2x.png"
                    else
                        saveStatusImage.source = "qrc:/qml/img/recycle-discard@2x.png"

                    if (index >= 0)
                        transactions.get(index).saveStatus = saveStatus
                }

                MouseArea {
                    id: statusMouseArea
                    anchors.fill: parent
                    onClicked:
                    {
                        parent.saveStatus = !parent.saveStatus
                    }
                }
            }

            Rectangle
            {
                Layout.preferredWidth: blockWidth
                Layout.preferredHeight: parent.height
                color: "#DEDCDC"
                RowLayout
                {
                    anchors.verticalCenter: parent.verticalCenter
                    spacing: cellSpacing
                    Text
                    {
                        id: hash
                        anchors.left: parent.left
                        anchors.leftMargin: horizontalMargin
                        Layout.preferredWidth: fromWidth
                        elide: Text.ElideRight
                        maximumLineCount: 1
                        text: {
                            if (index >= 0)
                                return transactions.get(index).sender
                            else
                                return ""
                        }
                    }

                    Text
                    {
                        id: func
                        text: {
                            if (index >= 0)
                                parent.userFrienldyToken(transactions.get(index).label)
                            else
                                return ""
                        }
                        elide: Text.ElideRight
                        maximumLineCount: 1
                        Layout.preferredWidth: toWidth
                    }

                    function userFrienldyToken(value)
                    {
                        if (value && value.indexOf("<") === 0)
                            return value.split(" - ")[0].replace("<", "") + "." + value.split("> ")[1] + "()";
                        else
                            return value
                    }

                    Text
                    {
                        id: returnValue
                        elide: Text.ElideRight
                        maximumLineCount: 1
                        Layout.preferredWidth: valueWidth
                        text: {
                            if (index >= 0 && transactions.get(index).returned)
                                return transactions.get(index).returned
                            else
                                return ""
                        }
                    }

                    Rectangle
                    {
                        Layout.preferredWidth: logsWidth
                        Layout.preferredHeight: trHeight - 10
                        width: logsWidth
                        color: "transparent"
                        Text
                        {
                            id: logs
                            anchors.left: parent.left
                            anchors.leftMargin: 10
                            text: {
                                if (index >= 0 && transactions.get(index).logs && transactions.get(index).logs.count)
                                {
                                    for (var k = 0; k < transactions.get(index).logs.count; k++)
                                    {
                                        /*console.log("_________________________")
                                        console.log(JSON.stringify(transactions.get(index).logs[k]))
                                        console.log("_________________________")*/
                                    }
                                    return transactions.get(index).logs.count
                                }
                                else
                                    return ""
                            }
                        }
                    }

                    Button
                    {
                        id: debug
                        Layout.preferredWidth: debugActionWidth
                        text: "debug"
                        onClicked:
                        {
                            clientModel.debugRecord(transactions.get(index).recordIndex);
                        }
                    }
                }
            }
        }
    }
}

