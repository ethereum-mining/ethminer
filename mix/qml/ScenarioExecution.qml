import QtQuick 2.2
import QtQuick.Controls 1.1
import QtQuick.Controls.Styles 1.1
import QtQuick.Dialogs 1.1
import QtQuick.Layouts 1.1
import Qt.labs.settings 1.0
import "js/Debugger.js" as Debugger
import "js/ErrorLocationFormater.js" as ErrorLocationFormater
import "."

Rectangle {

    border.color: "red"
    border.width: 1

    Connections
    {
        target:  projectModel
        onProjectLoaded: {
            loader.init()
        }

    }

    Column
    {
        anchors.margins: 10
        anchors.fill: parent
        spacing: 5
        ScenarioLoader
        {
            width: parent.width
            id: loader
        }

        Rectangle
        {
            width: parent.width
            height: 1
            color: "#cccccc"
        }

        Connections
        {
            target: loader
            onLoaded: {
                blockChain.load(scenario)
            }
        }

        BlockChain
        {
            id: blockChain
            width: parent.width
        }
    }
}
