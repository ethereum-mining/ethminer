import QtQuick 2.2
import QtQuick.Controls 1.1
import QtQuick.Layouts 1.0
import QtQuick.Controls.Styles 1.1
import CodeEditorExtensionManager 1.0
import QtWebEngine 1.0

Component {
	Item {
		signal editorTextChanged
		property string currentText: ""
		property bool initialized: false

		function setText(text) {
			currentText = text;
			if (initialized)
				editorBrowser.runJavaScript("setTextBase64(\"" + Qt.btoa(text) + "\")");
			editorBrowser.forceActiveFocus();
		}

		function getText() {
			return currentText;
		}

		anchors.top: parent.top
		id: codeEditorView
		anchors.fill: parent
		WebEngineView {
			id: editorBrowser
			url: "qrc:///qml/html/codeeditor.html"
			anchors.fill: parent
			onJavaScriptConsoleMessage:  {
				console.log("editor: " + sourceID + ":" + lineNumber + ":" + message);
			}

			onLoadingChanged:
			{
				if (!loading) {
					initialized = true;
					setText(currentText);
					runJavaScript("getTextChanged()", function(result) { });
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
							editorBrowser.runJavaScript("getText()" , function(textValue) {
								currentText = textValue;
								editorTextChanged();
							});
						}
					});
				}
			}
		}
	}
}
