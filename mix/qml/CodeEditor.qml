import QtQuick 2.2
import QtQuick.Controls 1.1
import QtQuick.Layouts 1.0
import QtQuick.Controls.Styles 1.1
import CodeEditorExtensionManager 1.0
import QtWebEngine 1.0
import QtWebEngine.experimental 1.0
import QtWebChannel 1.0

Rectangle {

	property alias textEditor: editorBrowser;

	anchors.top: parent.top
	id: codeEditorView
	width: parent.width
	height: parent.height * 0.7


	QtObject {
		id: textModel

		// the identifier under which this object
		// will be known on the JavaScript side
		WebChannel.id: "foo"

		// signals, methods and properties are
		// accessible to JavaScript code
		signal someSignal(string message);

		function someMethod(message) {
			console.log(message);
			someSignal(message);
			return "foobar";
		}

		property string hello: "world"
	}

	Rectangle {
		anchors.fill: parent


		WebEngineView {


			id: editorBrowser
			url: "qrc:///qml/html/codeeditor.html"
			//experimental.webChannel.registeredObjects: [myObject]
			experimental.inspectable: true

			anchors.fill: parent
			onJavaScriptConsoleMessage:  {

				console.log(sourceID + ":" + lineNumber + ":" + message);
			}


			onLoadingChanged:
			{
				console.log("onLoadingChanged");
				if (!loading) {
					console.log("onLoadingChangedDone");
					runJavaScript("getTextChanged()", function(result) {
						console.log(result);
					});
					pollTimer.running = true;
				}
			}

			Timer
			{
				id: pollTimer
				interval: 30
				running: false
				repeat: true
				onTriggered: {
					editorBrowser.runJavaScript("getTextChanged()", function(result) {
						if (result === true) {
							var textValue = editorBrowser.runJavaScript("getText()" , function(textValue) {
								codeModel.registerCodeChange(textValue);
							});
						}
					});
				}


			}
		}
	}
	/*
	Item {
			anchors.fill: parent
			visible: false
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
			font.pointSize: 12
			width: parent.width
			//anchors.centerIn: parent
			tabChangesFocus: false
			Keys.onPressed: {
				if (event.key === Qt.Key_Tab) {
					codeEditor.insert(codeEditor.cursorPosition, "\t");
					event.accepted = true;
				}
			}
		}
	}
	*/
}
