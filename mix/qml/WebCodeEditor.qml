import QtQuick 2.2
import QtQuick.Controls 1.1
import QtQuick.Layouts 1.0
import QtQuick.Controls.Styles 1.1
import QtWebEngine 1.0
import QtWebEngine.experimental 1.0
import org.ethereum.qml.Clipboard 1.0
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
	property var currentBreakpoints: []
	property string sourceName
	property var document
	property int fontSize: 0

	function setText(text, mode) {
		currentText = text;
		if (mode !== undefined)
			currentMode = mode;
		if (initialized && editorBrowser) {
			editorBrowser.runJavaScript("setTextBase64(\"" + Qt.btoa(text) + "\")");
			editorBrowser.runJavaScript("setMode(\"" + currentMode + "\")");
		}
		setFocus();
	}

	function setFocus() {
		if (editorBrowser)
			editorBrowser.forceActiveFocus();
	}

	function getText() {
		return currentText;
	}

	function syncClipboard() {
		if (Qt.platform.os == "osx" && editorBrowser) {
			var text = clipboard.text;
			editorBrowser.runJavaScript("setClipboardBase64(\"" + Qt.btoa(text) + "\")");
		}
	}

	function highlightExecution(location) {
		if (initialized && editorBrowser)
			editorBrowser.runJavaScript("highlightExecution(" + location.start + "," + location.end + ")");
	}

	function showWarning(content) {
		if (initialized && editorBrowser)
			editorBrowser.runJavaScript("showWarning('" + content + "')");
	}

	function getBreakpoints() {
		return currentBreakpoints;
	}

	function toggleBreakpoint() {
		if (initialized && editorBrowser)
			editorBrowser.runJavaScript("toggleBreakpoint()");
	}

	function changeGeneration() {
		if (initialized && editorBrowser)
			editorBrowser.runJavaScript("changeGeneration()", function(result) {});
	}

	function goToCompilationError() {
		if (initialized && editorBrowser)
			editorBrowser.runJavaScript("goToCompilationError()", function(result) {});
	}

	function setFontSize(size) {
		fontSize = size;
		if (initialized && editorBrowser)
			editorBrowser.runJavaScript("setFontSize(" + size + ")", function(result) {});
	}

	function setGasCosts(gasCosts) {
		if (initialized && editorBrowser)
			editorBrowser.runJavaScript("setGasCosts('" + JSON.stringify(gasCosts) + "')", function(result) {});
	}

	function displayGasEstimation(show) {
		if (initialized && editorBrowser)
			editorBrowser.runJavaScript("displayGasEstimation('" + show + "')", function(result) {});
	}

	Clipboard
	{
		id: clipboard
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

		Component.onDestruction:
		{
			codeModel.onCompilationComplete.disconnect(compilationComplete);
			codeModel.onCompilationError.disconnect(compilationError);
		}

		onLoadingChanged:
		{
			if (!loading && editorBrowser) {
				initialized = true;
				setFontSize(fontSize);
				setText(currentText, currentMode);
				runJavaScript("getTextChanged()", function(result) { });
				pollTimer.running = true;
				syncClipboard();
				if (currentMode === "solidity")
				{
					codeModel.onCompilationComplete.connect(compilationComplete);
					codeModel.onCompilationError.connect(compilationError);
				}
				parent.changeGeneration();
				loadComplete();
			}
		}


		function compilationComplete()
		{
			if (editorBrowser)
			{
				editorBrowser.runJavaScript("compilationComplete()", function(result) { });
				parent.displayGasEstimation(gasEstimationAction.checked);
			}


		}

		function compilationError(error, firstLocation, secondLocations)
		{
			if (!editorBrowser || !error)
				return;
			var detail = error.split('\n')[0];
			var reg = detail.match(/:\d+:\d+:/g);
			if (reg !== null)
				detail = detail.replace(reg[0], "");
			displayErrorAnnotations(detail, firstLocation, secondLocations);
		}

		function displayErrorAnnotations(detail, location, secondaryErrors)
		{
			editorBrowser.runJavaScript("compilationError('" + sourceName + "', '" + JSON.stringify(location) + "', '" + detail + "', '" + JSON.stringify(secondaryErrors) + "')", function(result){});
		}

		Timer
		{
			id: pollTimer
			interval: 30
			running: false
			repeat: true
			onTriggered: {
				if (!editorBrowser)
					return;
				editorBrowser.runJavaScript("getTextChanged()", function(result) {
					if (result === true && editorBrowser) {
						editorBrowser.runJavaScript("getText()" , function(textValue) {
							currentText = textValue;
							editorTextChanged();
						});
					}
				});
				editorBrowser.runJavaScript("getBreakpointsChanged()", function(result) {
					if (result === true && editorBrowser) {
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
