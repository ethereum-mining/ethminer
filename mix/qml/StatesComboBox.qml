/*
    This file is part of cpp-ethereum.

    cpp-ethereum is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    cpp-ethereum is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with cpp-ethereum.  If not, see <http://www.gnu.org/licenses/>.
*/
/** @file StatesComboBox.qml
 * @author Ali Mashatan ali@ethdev.com
 * @date 2015
 * Ethereum IDE client.
 */

import QtQuick 2.0
import QtQuick.Controls 1.0
import QtQuick.Layouts 1.1

Rectangle {
    id:statesComboBox

    width:200;
    height: 20;

    Component.onCompleted:
    {
        var top = dropDownList;
        while (top.parent)
        {
            top = top.parent
            if (top.objectName == "debugPanel")
                break;
        }
        var coordinates = dropDownList.mapToItem(top, 0, 0)
        dropDownList.parent = top;
        dropDownList.x = coordinates.x
        dropDownList.y = coordinates.y
    }

    signal selectItem(real item);
    signal editItem(real item);
    signal selectCreate();

    property variant items;
    readonly property alias selectedItem: chosenItemText.text;
    readonly property alias selectedIndex: listView.currentRow;
    function setSelectedIndex(index)
    {
        listView.currentRow = index;
        chosenItemText.text = statesComboBox.items.get(0).title;
    }

    signal comboClicked;

    property variant colorItem;
    property variant colorSelect;

    smooth:true;
    Rectangle {
        id:chosenItem
        width:parent.width;
        height:statesComboBox.height;
        color: statesComboBox.color;
        smooth:true;
        Text {
            id:chosenItemText
            anchors.top: parent.top;
            anchors.left: parent.left;
            anchors.margins: 2;
            color: statesComboBox.colorItem;
            text:""
            smooth:true
        }

        MouseArea {
            anchors.fill: parent;
            onClicked: {
                statesComboBox.state = statesComboBox.state==="dropDown"?"":"dropDown"
            }
        }
    }
//ToDo: We need scrollbar for items
    Rectangle {
        id:dropDownList
        width:statesComboBox.width;
        height:0;
        clip:true;
        radius:4;
        anchors.top: chosenItem.bottom;
        anchors.margins: 2;
        color: statesComboBox.color
        ColumnLayout{
            spacing: 2
            TableView {
                id:listView
                height:20;
                implicitHeight: 0
                width:statesComboBox.width;
                model: statesComboBox.items
                horizontalScrollBarPolicy: Qt.ScrollBarAlwaysOff;
                currentRow: -1
                headerVisible: false;
                backgroundVisible: false
                alternatingRowColors : false;
                frameVisible: false

                TableViewColumn {
                    role: "title"
                    title: ""
                    width: statesComboBox.width;
                    delegate: mainItemDelegate
                }

                Component {
                    id: mainItemDelegate
                    Item{
                        id: itemDelegate
                        width:statesComboBox.width;
                        height: statesComboBox.height;
                        Text {
                            id: textItemid
                            text: styleData.value
                            color: statesComboBox.colorItem;
                            anchors.top: parent.top;
                            anchors.left: parent.left;
                            anchors.margins: 5;

                        }
                        Rectangle
                        {
                            id: spaceItemid
                            anchors.top: textItemid.top;
                            anchors.left: textItemid.right
                            width: parent.width - textItemid.width - imageItemid.width - textItemid.anchors.margins- textItemid.anchors.margins
                        }
                        Image {
                            id: imageItemid
                            height:20
                            width:20;
                            visible: false;
                            fillMode: Image.PreserveAspectFit
                            source: "img/edit_combox.png"
                            anchors.top: spaceItemid.top;
                            anchors.left: spaceItemid.right;
                        }

                        MouseArea {
                            anchors.fill: parent;
                            hoverEnabled : true

                            onEntered: {
                                imageItemid.visible = true;
                                textItemid.color = statesComboBox.colorSelect;
                            }
                            onExited: {
                                imageItemid.visible = false;
                                textItemid.color = statesComboBox.colorItem;
                            }
                            onClicked: {
                                if (mouseX > imageItemid.x && mouseX < imageItemid.x+ imageItemid.width
                                        && mouseY > imageItemid.y && mouseY < imageItemid.y+ imageItemid.height)
                                    statesComboBox.editItem(styleData.row);
                                else {
                                    statesComboBox.state = ""
                                    var prevSelection = chosenItemText.text
                                    chosenItemText.text = styleData.value
                                    listView.currentRow = styleData.row;
                                    statesComboBox.selectItem(styleData.row);
                                }
                            }
                        }
                    }//Item
                }//Component
            }//Table View
            RowLayout{
                Rectangle{
                    width: 1
                }
                Text{
                    id:createStateText
                    width:statesComboBox.width;
                    height: statesComboBox.height;
                    font.pointSize: 10
                    text:"Create State ..."
                    MouseArea
                    {
                        anchors.fill: parent;
                        hoverEnabled : true

                        onEntered: {
                            createStateText.color = statesComboBox.colorSelect;
                        }
                        onExited: {
                            createStateText.color = statesComboBox.colorItem;
                        }
                        onClicked: {
                            statesComboBox.state = ""
                            statesComboBox.selectCreate();
                        }
                    }
                }
            }
        }
    }
    states: State {
        name: "dropDown";
        PropertyChanges { target: dropDownList; height:(20*(statesComboBox.items.count+1)) }
        PropertyChanges { target:listView; height:20; implicitHeight: (20*(statesComboBox.items.count)) }
    }

}
