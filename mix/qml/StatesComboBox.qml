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
import org.ethereum.qml.InverseMouseArea 1.0

Rectangle {
	id: statesComboBox

	width: 200
	height: 23
	Component.onCompleted: {
		var top = dropDownList
		while (top.parent) {
			top = top.parent
			if (top.objectName == "debugPanel")
				break
		}
		var coordinates = dropDownList.mapToItem(top, 0, 0)
		//the order is important
		dropDownShowdowList.parent = top
		dropDownList.parent = top

		dropDownShowdowList.x = coordinates.x
		dropDownShowdowList.y = coordinates.y

		dropDownList.x = coordinates.x
		dropDownList.y = coordinates.y
	}

	signal selectItem(real item)
	signal editItem(real item)
	signal selectCreate
	property int rowHeight: 25
	property variant items
	property alias selectedItem: chosenItemText.text
	property alias selectedIndex: listView.currentRow
	function setSelectedIndex(index) {
		listView.currentRow = index
		chosenItemText.text = statesComboBox.items.get(index).title
	}

	function hideDropDown()
	{
		if (statesComboBox.state === "dropDown")
			statesComboBox.state = "";
	}

	signal comboClicked

	property variant colorItem
	property variant colorSelect

	SourceSansProRegular
	{
		id: regularFont
	}

	SourceSansProBold
	{
		id: boldFont
	}
	smooth: true
	Rectangle {
		id: chosenItem
		width: parent.width
		height: statesComboBox.height
		color: statesComboBox.color
		Text {
			id: chosenItemText
			anchors.left: parent.left
			anchors.leftMargin: 10
			anchors.verticalCenter: parent.verticalCenter
			color: statesComboBox.colorItem
			text: ""
			font.family: regularFont.name
		}

		MouseArea {
			id: selectorArea
			anchors.fill: parent
			onClicked: {
				if (statesCombo.state === "")
					statesCombo.state = "dropDown";
			}
		}
	}

	Rectangle {
		id: dropDownShowdowList
		width: statesComboBox.width
		opacity: 0.3
		height: 0
		clip: true
		radius: 4
		anchors.top: chosenItem.top
		anchors.margins: 2
		color: "gray"
	}
	//ToDo: We need scrollbar for items
	Rectangle {
		id: dropDownList
		width: statesComboBox.width
		height: 0
		clip: true
		radius: 4
		anchors.top: chosenItem.top
		anchors.topMargin: 23
		color: statesComboBox.color

		InverseMouseArea
		{
			id: outsideClick
			anchors.fill: parent
			active: false
			onClickedOutside: {
				var p = selectorArea.mapFromItem(null, _point.x, _point.y);
				if (!selectorArea.contains(Qt.point(p.x, p.y)))
					statesCombo.hideDropDown()
			}
		}

		ColumnLayout {
			spacing: 2
			TableView {
				id: listView
				height: 20
				implicitHeight: 0
				width: statesComboBox.width
				model: statesComboBox.items
				horizontalScrollBarPolicy: Qt.ScrollBarAlwaysOff
				currentRow: -1
				headerVisible: false
				backgroundVisible: false
				alternatingRowColors: false
				frameVisible: false

				TableViewColumn {
					role: "title"
					title: ""
					width: statesComboBox.width
					delegate: mainItemDelegate
				}
				rowDelegate: Rectangle {
					width: statesComboBox.width
					height: statesComboBox.rowHeight
				}
				Component {
					id: mainItemDelegate
					Rectangle {
						id: itemDelegate
						width: statesComboBox.width
						height: statesComboBox.height
						Text {
							id: textItemid
							text: styleData.value
							color: statesComboBox.colorItem
							anchors.top: parent.top
							anchors.left: parent.left
							anchors.leftMargin: 10
							anchors.topMargin: 5
							font.family: regularFont.name
						}
						Image {
							id: imageItemid
							height: 20
							width: 20
							anchors.right: parent.right
							anchors.top: parent.top
							anchors.margins: 5
							visible: false
							fillMode: Image.PreserveAspectFit
							source: "img/edit_combox.png"
						}

						MouseArea {
							anchors.fill: parent
							hoverEnabled: true

							onEntered: {
								imageItemid.visible = true
								textItemid.color = statesComboBox.colorSelect
							}
							onExited: {
								imageItemid.visible = false
								textItemid.color = statesComboBox.colorItem
							}
							onClicked: {
								if (mouseX > imageItemid.x
										&& mouseX < imageItemid.x + imageItemid.width
										&& mouseY > imageItemid.y
										&& mouseY < imageItemid.y + imageItemid.height)
									statesComboBox.editItem(styleData.row)
								else {
									statesComboBox.state = ""
									var prevSelection = chosenItemText.text
									chosenItemText.text = styleData.value
									listView.currentRow = styleData.row
									statesComboBox.selectItem(styleData.row)
								}
							}
						}
					} //Item
				} //Component
			} //Table View

			RowLayout {
				anchors.top: listView.bottom
				anchors.topMargin: 4
				anchors.left: parent.left
				anchors.leftMargin: 10
				Text {
					id: createStateText
					width: statesComboBox.width
					height: statesComboBox.height
					font.family: boldFont.name
					color: "#808080"
					text: qsTr("Create State ...")
					font.weight: Font.DemiBold
					MouseArea {
						anchors.fill: parent
						hoverEnabled: true

						onEntered: {
							createStateText.color = statesComboBox.colorSelect
						}
						onExited: {
							createStateText.color = statesComboBox.colorItem
						}
						onClicked: {
							statesComboBox.state = ""
							statesComboBox.selectCreate()
						}
					}
				}
			}
		}
	}
	states: State {
		name: "dropDown"
		PropertyChanges {
			target: dropDownList
			height: (statesComboBox.rowHeight * (statesComboBox.items.count + 1))
		}
		PropertyChanges {
			target: dropDownShowdowList
			width: statesComboBox.width + 3
			height: (statesComboBox.rowHeight * (statesComboBox.items.count + 1)) + 3
		}
		PropertyChanges {
			target: listView
			height: 20
			implicitHeight: (statesComboBox.rowHeight * (statesComboBox.items.count))
		}
		PropertyChanges {
			target: outsideClick
			active: true
		}
	}
}
