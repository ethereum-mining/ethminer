import QtQuick 2.2
import QtQuick.Controls 1.1
import QtQuick.Controls.Styles 1.1
import QtQuick.Dialogs 1.1
import QtQuick.Layouts 1.1
import Qt.labs.settings 1.0
import "js/Debugger.js" as Debugger
import "js/ErrorLocationFormater.js" as ErrorLocationFormater
import "."

RowLayout
{
    signal restored(variant scenario)
    signal saved(variant scenario)
    signal duplicated(variant scenario)
    signal loaded(variant scenario)

    function init()
    {
        scenarioList.load()
    }

    id: blockChainSelector
    ComboBox
    {
        id: scenarioList
        model: projectModel.stateListModel
        textRole: "title"
        onCurrentIndexChanged:
        {
            restoreScenario.restore()
        }

        function load()
        {
            var state = projectModel.stateListModel.getState(currentIndex)
            loaded(state)
        }
    }

    ScenarioButton {
        id: restoreScenario
        Layout.preferredWidth: 100
        Layout.preferredHeight: 30
        buttonShortcut: ""
        sourceImg: "qrc:/qml/img/restoreIcon@2x.png"
        onClicked: {
            restore()
        }
        text: qsTr("Restore")
        function restore()
        {
            var state = projectModel.stateListModel.reloadStateFromFromProject(scenarioList.currentIndex)
            restored(state)
            loaded(state)
        }
    }

    ScenarioButton {
        id: saveScenario
        text: qsTr("Save")
        onClicked: {
            projectModel.saveProjectFile()
            saved(state)
        }
        Layout.preferredWidth: 100
        Layout.preferredHeight: 30
        buttonShortcut: ""
        sourceImg: "qrc:/qml/img/saveIcon@2x.png"
    }

    ScenarioButton
    {
        id: duplicateScenario
        text: qsTr("Duplicate")
        onClicked: {
            var state = JSON.parse(JSON.stringify(projectModel.stateListModel.getState(scenarioList.currentIndex)))
            state.title = qsTr("Copy of ") + state.title;
            projectModel.stateListModel.appendState(state)
            projectModel.stateListModel.save()
            duplicated(state)
        }
        Layout.preferredWidth: 100
        Layout.preferredHeight: 30
        buttonShortcut: ""
        sourceImg: "qrc:/qml/img/duplicateIcon@2x.png"
    }
}
