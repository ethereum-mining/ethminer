import QtQuick 2.2
import QtQuick.Controls 1.1
import QtQuick.Layouts 1.0
import QtQuick.Controls.Styles 1.1
import QtWebEngine 1.0
import QtWebEngine.experimental 1.0

Item {
	signal editorTextChanged
	signal breakpointsChanged
	property bool isClean: true
	property string currentText: ""
	property string currentMode: ""
	property bool initialized: false
	property var currentBreakpoints: [];

	function setText(text, mode) {
		currentText = text;
		currentMode = mode;
		if (initialized) {
			editorBrowser.runJavaScript("setTextBase64(\"" + Qt.btoa(text) + "\")");
			editorBrowser.runJavaScript("setMode(\"" + mode + "\")");
		}
		setFocus();
	}

	function setFocus() {
		editorBrowser.forceActiveFocus();
	}

	function getText() {
		return currentText;
	}

	function syncClipboard() {
		if (Qt.platform.os == "osx") {
			var text = appContext.clipboard;
			editorBrowser.runJavaScript("setClipboardBase64(\"" + Qt.btoa(text) + "\")");
		}
	}

	function highlightExecution(location) {
		if (initialized)
			editorBrowser.runJavaScript("highlightExecution(" + location.start + "," + location.end + ")");
	}

	function getBreakpoints() {
		return currentBreakpoints;
	}

	function toggleBreakpoint() {
		if (initialized)
			editorBrowser.runJavaScript("toggleBreakpoint()");
	}

	function changeGeneration() {
		if (initialized)
			editorBrowser.runJavaScript("changeGeneration()", function(result) {});
	}

	Connections {
		target: appContext
		onClipboardChanged:	syncClipboard()
	}

	anchors.top: parent.top
	id: codeEditorView
	anchors.fill: parent
	WebEngineView {
		id: editorBrowser
		url: "qrc:///qml/html/codeeditor.html"
		anchors.fill: parent
		experimental.settings.javascriptCanAccessClipboard: true
		onJavaScriptConsoleMessage:  {
			console.log("editor: " + sourceID + ":" + lineNumber + ":" + message);
		}

		onLoadingChanged:
		{
			if (!loading) {
				initialized = true;
				setText(currentText, currentMode);
				runJavaScript("getTextChanged()", function(result) { });
				pollTimer.running = true;
				syncClipboard();
				parent.changeGeneration();
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
				editorBrowser.runJavaScript("getBreakpointsChanged()", function(result) {
					if (result === true) {
						editorBrowser.runJavaScript("getBreakpoints()" , function(bp) {
							if (currentBreakpoints !== bp) {
								currentBreakpoints = bp;
								breakpointsChanged();
							}
						});
					}
				});
				editorBrowser.runJavaScript("isClean()", function(result) {
					isClean = result;
				});
			}
		}
	}
}
