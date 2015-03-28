import QtQuick 2.2
import QtQuick.Controls 1.1
import QtQuick.Layouts 1.0
import QtQuick.Controls.Styles 1.1
import QtWebEngine 1.0
import QtWebEngine.experimental 1.0
import "js/ErrorLocationFormater.js" as ErrorLocationFormater

Item {
	signal breakpointsChanged
	signal editorTextChanged
	signal loadComplete
	property bool isClean: true
	property string currentText: ""
	property string currentMode: ""
	property bool initialized: false
	property bool unloaded: false
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
			var text = clipboard.text;
			editorBrowser.runJavaScript("setClipboardBase64(\"" + Qt.btoa(text) + "\")");
		}
	}

	function highlightExecution(location) {
		if (initialized)
			editorBrowser.runJavaScript("highlightExecution(" + location.start + "," + location.end + ")");
	}

	function showWarning(content) {
		if (initialized)
			editorBrowser.runJavaScript("showWarning('" + content + "')");
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
		target: clipboard
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
			if (!loading && !unloaded && editorBrowser) {
				initialized = true;
				setText(currentText, currentMode);
				runJavaScript("getTextChanged()", function(result) { });
				pollTimer.running = true;
				syncClipboard();
				if (currentMode === "solidity")
				{
					codeModel.onCompilationComplete.connect(function(){
						editorBrowser.runJavaScript("compilationComplete()", function(result) { });
					});

					codeModel.onCompilationError.connect(function(error){
						var errorInfo = ErrorLocationFormater.extractErrorInfo(error, false);
						if (errorInfo.line && errorInfo.column)
							editorBrowser.runJavaScript("compilationError('" +  errorInfo.line + "', '" +  errorInfo.column + "', '" +  errorInfo.errorDetail + "')", function(result) { });
						else
							editorBrowser.runJavaScript("compilationComplete()", function(result) { });
					});
				}
				parent.changeGeneration();
				loadComplete();
			}
		}

		Timer
		{
			id: pollTimer
			interval: 30
			running: false
			repeat: true
			onTriggered: {
				if (unloaded)
					return;
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
