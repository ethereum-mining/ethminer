import QtQuick 2.0
import QtQuick.Controls 1.3

Item
{
	property alias value: textinput.text
	property alias contractCreationTr: ctrModel
	id: editRoot
	height: 30
	width: 200

	SourceSansProBold
	{
		id: boldFont
	}

	function init()
	{
		if (value.indexOf("<") === 0)
		{
			for (var k = 0; k < ctrModel.count; k++)
			{
				if ("<" + ctrModel.get(k).functionId + ">" === value)
				{
					trCombobox.currentIndex = k;
					return;
				}
			}
			trCombobox.currentIndex = 0;
			value = "";
		}
	}

	Rectangle {
		anchors.fill: parent
		radius: 4
		TextInput {
			id: textinput
			text: value
			width: parent.width
			height: 30
			wrapMode: Text.WrapAnywhere
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
		textRole: "functionId"
		anchors.verticalCenter: parent.verticalCenter
		anchors.left: textinput.parent.right
		onCurrentIndexChanged: {
			if (currentText === "")
				return;
			else if (currentText !== " - ")
			{
				textinput.text = "<" + currentText + ">";
				trCombobox.selected = true;
			}
			else if (textinput.text.indexOf("<") === 0)
				textinput.text = "";
		}
	}
}
