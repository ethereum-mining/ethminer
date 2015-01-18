import QtQuick 2.2
import QtQuick.Controls 1.1
import QtQuick.Layouts 1.1
import QtQuick.Controls.Styles 1.1

Rectangle {
	id: etherEdition
	property bool displayFormattedValue;
	property bool edit;
	property variant value;
	onValueChanged: update()
	Component.onCompleted: update()

	function update()
	{
		etherValueEdit.text = value.value;
		selectUnit(value.unit);
	}

	function selectUnit(unit)
	{
		for(var i = 0; i < unitsModel.count; ++i)
		{
			if (unitsModel.get(i).text === unit)
			{
				units.currentIndex = i;
				return;
			}
		}
	}

	RowLayout
	{
		anchors.fill: parent;
		id: row
		width: 200
		height: parent.height
		Rectangle
		{
			width : 200
			color: edit ? "blue" : "white"
			TextField
			{
				onTextChanged:
				{
					value.setValue(text)
					formattedValue.text = value.format();
				}
				width: parent.width
				readOnly: !edit
				visible: edit
				id: etherValueEdit;
			}
		}

		Rectangle
		{
			Layout.fillWidth: true
			id: unitContainer
			width: 20
			anchors.verticalCenter: parent.verticalCenter
			ComboBox
			{
				id: units
				onCurrentTextChanged:
				{
					value.setUnit(currentText);
					formattedValue.text = value.format();
				}
				model: ListModel {
					id: unitsModel
					ListElement { text: "wei"; }
					ListElement { text: "Kwei"; }
					ListElement { text: "Mwei"; }
					ListElement { text: "Gwei"; }
					ListElement { text: "szabo"; }
					ListElement { text: "finney"; }
					ListElement { text: "ether"; }
					ListElement { text: "grand"; }
					ListElement { text: "Mether"; }
					ListElement { text: "Gether"; }
					ListElement { text: "Tether"; }
					ListElement { text: "Pether"; }
					ListElement { text: "Eether"; }
					ListElement { text: "Zether"; }
					ListElement { text: "Yether"; }
					ListElement { text: "Nether"; }
					ListElement { text: "Dether"; }
					ListElement { text: "Vether"; }
					ListElement { text: "Uether"; }
				}
			}
			Rectangle
			{
				anchors.verticalCenter: parent.verticalCenter
				anchors.left: units.right
				visible: displayFormattedValue
				width: 20
				Text
				{
					id: formattedValue
				}
			}
		}
	}
}
