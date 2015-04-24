import QtQuick 2.0
import QtQuick.Controls 1.3
import QtQuick.Controls.Styles 1.3

Item
{
	property alias value: textinput.text
	property alias accountRef: ctrModel
	property string subType
	property bool readOnly
	id: editRoot
	height: 20
	width: 320

	SourceSansProBold
	{
		id: boldFont
	}

	function init()
	{
		trCombobox.visible = !readOnly
		textinput.readOnly = readOnly
		if (!readOnly)
		{
			for (var k = 0; k < ctrModel.count; k++)
			{
				if (ctrModel.get(k).value === value)
				{
					trCombobox.currentIndex = k;
					return;
				}
			}
			trCombobox.currentIndex = 0;
		}
	}

	Rectangle {
		anchors.fill: parent
		radius: 4
		anchors.verticalCenter: parent.verticalCenter
		height: 20
		TextInput {
			id: textinput
			text: value
			width: parent.width
			height: parent.width
			wrapMode: Text.WordWrap
			clip: true
			font.family: boldFont.name
			MouseArea {
				id: mouseArea
				anchors.fill: parent
				hoverEnabled: true
				onClicked: textinput.forceActiveFocus()
			}
			onTextChanged:
			{
				if (trCombobox.selected)
				{
					trCombobox.currentIndex = 0;
					trCombobox.selected = false;
				}
			}
		}
	}

	ListModel
	{
		id: ctrModel
	}

	ComboBox
	{
		property bool selected: false
		id: trCombobox
		model: ctrModel
		textRole: "itemid"
		height: 20
		anchors.verticalCenter: parent.verticalCenter
		anchors.left: textinput.parent.right
		anchors.leftMargin: 3
		onCurrentIndexChanged: {
			trCombobox.selected = false;
			if (currentText === "")
				return;
			else if (currentText !== " - ")
			{
				if (model.get(currentIndex).type === "contract")
					textinput.text = "<" + currentText + ">";
				else
					textinput.text = model.get(currentIndex).value; //address
				trCombobox.selected = true;
			}
			else if (textinput.text.indexOf("<") === 0)
			{
				textinput.text = "";
			}
		}
	}
}
