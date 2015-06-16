import QtQuick 2.0

Item
{
	property string value
	property alias readOnly: textinput.readOnly
	id: editRoot
	height: 20
	width: readOnly ? textinput.implicitWidth : 150
	onValueChanged:
	{
		 textinput.text = value
	}

	SourceSansProBold
	{
		id: boldFont
	}

	Rectangle {
		anchors.fill: parent
		radius: 4
		TextInput {
			id: textinput
			clip: true
			anchors.fill: parent
			wrapMode: Text.WrapAnywhere
			font.family: boldFont.name
			selectByMouse: true
			onTextChanged: {
				var stringRegEx = new RegExp('"^\\"*', "g")
				var str = stringRegEx.exec(text)
				if (str && str.length > 0)
					value = str[0]
				else
					value = text
			}
		}
	}
}



