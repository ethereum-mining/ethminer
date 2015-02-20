import QtQuick 2.0
import QtQuick.Window 2.0
import QtQuick.Layouts 1.0
import QtQuick.Controls 1.0
import QtQuick.Controls.Styles 1.1
import "."

Item {
	signal editorTextChanged

	function setText(text) {
		codeEditor.text = text;
	}

	function getText() {
		return codeEditor.text;
	}

	function setFocus() {
		codeEditor.forceActiveFocus();
	}

	anchors.fill: parent
	id: contentView
	width: parent.width
	height: parent.height * 0.7
		Rectangle {
			id: lineColumn
			property int rowHeight: codeEditor.font.pixelSize + 3
			color: "#202020"
			width: 50
			height: parent.height
			Column {
				y: -codeEditor.flickableItem.contentY + 4
				width: parent.width
				Repeater {
					model: Math.max(codeEditor.lineCount + 2, (lineColumn.height/lineColumn.rowHeight))
					delegate: Text {
						id: text
						color: codeEditor.textColor
						font: codeEditor.font
						width: lineColumn.width - 4
						horizontalAlignment: Text.AlignRight
						verticalAlignment: Text.AlignVCenter
						height: lineColumn.rowHeight
						renderType: Text.NativeRendering
						text: index + 1
					}
				}
			}
		}

	TextArea {
		id: codeEditor
		textColor: "#EEE8D5"
		style: TextAreaStyle {
			backgroundColor: "#002B36"
		}

		anchors.left: lineColumn.right
		anchors.right: parent.right
		anchors.top: parent.top
		anchors.bottom: parent.bottom
		wrapMode: TextEdit.NoWrap
		frameVisible: false

		height: parent.height
		font.family: "Monospace"
		font.pointSize: CodeEditorStyle.general.basicFontSize
		width: parent.width

		tabChangesFocus: false
		Keys.onPressed: {
			if (event.key === Qt.Key_Tab) {
				codeEditor.insert(codeEditor.cursorPosition, "\t");
				event.accepted = true;
			}
		}
		onTextChanged: {
			editorTextChanged();
		}

	}
}
