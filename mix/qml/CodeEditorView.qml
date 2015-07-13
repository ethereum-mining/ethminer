import QtQuick 2.0
import QtQuick.Window 2.0
import QtQuick.Layouts 1.0
import QtQuick.Controls 1.0
import QtQuick.Dialogs 1.1
import Qt.labs.settings 1.0

Item {
	id: codeEditorView
	property string currentDocumentId: ""
	property string sourceInError
	property int openDocCount: 0
	signal documentEdit(string documentId)
	signal breakpointsChanged(string documentId)
	signal isCleanChanged(var isClean, string documentId)
	signal loadComplete

	function getDocumentText(documentId) {
		for (var i = 0; i < openDocCount; i++)	{
			if (editorListModel.get(i).documentId === documentId) {
				return editors.itemAt(i).item.getText();
			}
		}
		return "";
	}

	function getContracts()
	{
		var ctr = []
		for (var i = 0; i < openDocCount; i++)
		{
			if (editorListModel.get(i).isContract)
			{
				ctr.push(editors.itemAt(i).item)
			}
		}
		return ctr;
	}

	function isDocumentOpen(documentId) {
		for (var i = 0; i < openDocCount; i++)
			if (editorListModel.get(i).documentId === documentId &&
					editors.itemAt(i).item)
				return true;
		return false;
	}

	function openDocument(document)	{
		loadDocument(document);
		currentDocumentId = document.documentId;
	}

	function loadDocument(document) {
		for (var i = 0; i < openDocCount; i++)
			if (editorListModel.get(i).documentId === document.documentId)
				return; //already open

		if (editorListModel.count <= openDocCount)
			editorListModel.append(document);
		else
		{
			editorListModel.set(openDocCount, document);
			doLoadDocument(editors.itemAt(openDocCount).item, editorListModel.get(openDocCount), false)
			loadComplete();
		}
		openDocCount++;
	}

	function doLoadDocument(editor, document, create) {
		var data = fileIo.readFile(document.path);
		if (create)
		{
			editor.onLoadComplete.connect(function() {
				codeEditorView.loadComplete();
			});
			editor.onEditorTextChanged.connect(function() {
				documentEdit(editor.document.documentId);
				if (editor.document.isContract)
					codeModel.registerCodeChange(editor.document.documentId, editor.getText());
			});
			editor.onBreakpointsChanged.connect(function() {
				if (editor.document.isContract)
					breakpointsChanged(editor.document.documentId);
			});
			editor.onIsCleanChanged.connect(function() {
				isCleanChanged(editor.isClean, editor.document.documentId);
			});
		}
		editor.document = document;
		editor.setFontSize(editorSettings.fontSize);
		editor.sourceName = document.documentId;
		editor.setText(data, document.syntaxMode);
		editor.changeGeneration();
	}

	function getEditor(documentId) {
		for (var i = 0; i < openDocCount; i++)
		{
			if (editorListModel.get(i).documentId === documentId)
				return editors.itemAt(i).item;
		}
		return null;
	}

	function highlightExecution(documentId, location)
	{
		var editor = getEditor(documentId);
		if (editor)
		{
			if (documentId !== location.sourceName)
				findAndHightlight(location.start, location.end, location.sourceName)
			else
				editor.highlightExecution(location);
		}
	}

	// Execution is not in the current document. Try:
	// Open targeted document and hightlight (TODO) or
	// Warn user that file is not available
	function findAndHightlight(start, end, sourceName)
	{
		var editor = getEditor(currentDocumentId);
		if (editor)
			editor.showWarning(qsTr("Currently debugging in " + sourceName + ". Source not available."));
	}

	function editingContract() {
		for (var i = 0; i < openDocCount; i++)
			if (editorListModel.get(i).documentId === currentDocumentId)
				return editorListModel.get(i).isContract;
		return false;
	}

	function getBreakpoints() {
		var bpMap = {};
		for (var i = 0; i < openDocCount; i++)  {
			var documentId = editorListModel.get(i).documentId;
			var editor = editors.itemAt(i).item;
			if (editor) {
				bpMap[documentId] = editor.getBreakpoints();
			}
		}
		return bpMap;
	}

	function toggleBreakpoint() {
		var editor = getEditor(currentDocumentId);
		if (editor)
			editor.toggleBreakpoint();
	}

	function resetEditStatus(docId) {
		var editor = getEditor(docId);
		if (editor)
			editor.changeGeneration();
	}

	function goToCompilationError() {
		if (sourceInError === "")
			return;
		if (currentDocumentId !== sourceInError)
			projectModel.openDocument(sourceInError);
		for (var i = 0; i < openDocCount; i++)
		{
			var doc = editorListModel.get(i);
			if (doc.isContract && doc.documentId === sourceInError)
			{
				var editor = editors.itemAt(i).item;
				if (editor)
					editor.goToCompilationError();
				break;
			}
		}
	}

	function setFontSize(size) {
		if (size <= 10 || size >= 48)
			return;
		editorSettings.fontSize = size;
		for (var i = 0; i < editors.count; i++)
			editors.itemAt(i).item.setFontSize(size);
	}

	function displayGasEstimation(checked)
	{
		var editor = getEditor(currentDocumentId);
		if (editor)
			editor.displayGasEstimation(checked);
	}

	Component.onCompleted: projectModel.codeEditor = codeEditorView;

	Connections {
		target: codeModel
		onCompilationError: {
			sourceInError = _firstErrorLoc.source;
		}
		onCompilationComplete: {
			sourceInError = "";
			var gasCosts = codeModel.gasCostByDocumentId(currentDocumentId);
			var editor = getEditor(currentDocumentId);
			if (editor)
				editor.setGasCosts(gasCosts);
		}
	}

	Connections {
		target: projectModel
		onDocumentOpened: {
			openDocument(document);
		}

		onProjectSaving: {
			for (var i = 0; i < openDocCount; i++)
			{
				var doc = editorListModel.get(i);
				if (editors.itemAt(i))
				{
					var editor = editors.itemAt(i).item;
					if (editor)
						fileIo.writeFile(doc.path, editor.getText());
				}
			}
		}

		onProjectSaved: {
			if (projectModel.appIsClosing || projectModel.projectIsClosing)
				return;
			for (var i = 0; i < openDocCount; i++)
			{
				var doc = editorListModel.get(i);
				resetEditStatus(doc.documentId);
			}
		}

		onProjectClosed: {
			currentDocumentId = "";
			openDocCount = 0;
		}

		onDocumentSaved: {
			resetEditStatus(documentId);
		}

		onContractSaved: {
			resetEditStatus(documentId);
		}

		onDocumentSaving: {
			for (var i = 0; i < editorListModel.count; i++)
			{
				var doc = editorListModel.get(i);
				if (doc.path === document.path)
				{
					fileIo.writeFile(document.path, editors.itemAt(i).item.getText());
					break;
				}
			}
		}
	}

	CodeEditorStyle
	{
		id: style;
	}

	MessageDialog
	{
		id: messageDialog
		title: qsTr("File Changed")
		text: qsTr("This file has been changed outside of the editor. Do you want to reload it?")
		standardButtons: StandardButton.Yes | StandardButton.No
		property variant item
		property variant doc
		onYes: {
			doLoadDocument(item, doc, false);
			resetEditStatus(doc.documentId);
		}
	}

	Repeater {
		id: editors
		model: editorListModel
		onItemRemoved: {
			item.item.unloaded = true;
		}
		delegate: Loader {
			id: loader
			active: false
			asynchronous: true
			anchors.fill:  parent
			source: appService.haveWebEngine ? "WebCodeEditor.qml" : "CodeEditor.qml"
			visible: (index >= 0 && index < openDocCount && currentDocumentId === editorListModel.get(index).documentId)
			property bool changed: false
			onVisibleChanged: {
				loadIfNotLoaded()
				if (visible && item)
				{
					loader.item.setFocus();
					if (changed)
					{
						changed = false;
						messageDialog.item = loader.item;
						messageDialog.doc = editorListModel.get(index);
						messageDialog.open();
					}
					loader.item.displayGasEstimation(gasEstimationAction.checked);
				}
			}
			Component.onCompleted: {
				loadIfNotLoaded()
			}
			onLoaded: {
				doLoadDocument(loader.item, editorListModel.get(index), true)
			}

			Connections
			{
				target: projectModel
				onDocumentChanged: {
					if (!item)
						return;
					var current = editorListModel.get(index);
					if (documentId === current.documentId)
					{
						if (currentDocumentId === current.documentId)
						{
							messageDialog.item = loader.item;
							messageDialog.doc = editorListModel.get(index);
							messageDialog.open();
						}
						else
							changed = true
					}
				}

				onDocumentUpdated: {
					var document = projectModel.getDocument(documentId);
					for (var i = 0; i < editorListModel.count; i++)
						if (editorListModel.get(i).documentId === documentId)
						{
							editorListModel.set(i, document);
							break;
						}
				}

				onDocumentRemoved: {
					for (var i = 0; i < editorListModel.count; i++)
						if (editorListModel.get(i).documentId === documentId)
						{
							editorListModel.remove(i);
							openDocCount--;
							break;
						}
				}
			}

			function loadIfNotLoaded () {
				if (visible && !active) {
					active = true;
				}
			}
		}
	}
	ListModel {
		id: editorListModel
	}

	Action {
		id: increaseFontSize
		text: qsTr("Increase Font Size")
		shortcut: "Ctrl+="
		onTriggered: setFontSize(editorSettings.fontSize + 1)
	}

	Action {
		id: decreaseFontSize
		text: qsTr("Decrease Font Size")
		shortcut: "Ctrl+-"
		onTriggered: setFontSize(editorSettings.fontSize - 1)
	}

	Settings {
		id: editorSettings
		property int fontSize: 12;
	}
}
