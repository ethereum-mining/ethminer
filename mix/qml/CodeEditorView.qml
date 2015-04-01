import QtQuick 2.0
import QtQuick.Window 2.0
import QtQuick.Layouts 1.0
import QtQuick.Controls 1.0
import QtQuick.Dialogs 1.1

Item {
	id: codeEditorView
	property string currentDocumentId: ""
	signal documentEdit(string documentId)
	signal breakpointsChanged(string documentId)
	signal isCleanChanged(var isClean, string documentId)


	function getDocumentText(documentId) {
		for (var i = 0; i < editorListModel.count; i++)	{
			if (editorListModel.get(i).documentId === documentId) {
				return editors.itemAt(i).item.getText();
			}
		}
		return "";
	}

	function isDocumentOpen(documentId) {
		for (var i = 0; i < editorListModel.count; i++)
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
		for (var i = 0; i < editorListModel.count; i++)
			if (editorListModel.get(i).documentId === document.documentId)
				return; //already open

		editorListModel.append(document);
	}

	function doLoadDocument(editor, document) {
		var data = fileIo.readFile(document.path);
		editor.onEditorTextChanged.connect(function() {
			documentEdit(document.documentId);
			if (document.isContract)
				codeModel.registerCodeChange(document.documentId, editor.getText());
		});
		editor.onBreakpointsChanged.connect(function() {
			if (document.isContract)
				breakpointsChanged(document.documentId);
		});
		editor.setText(data, document.syntaxMode);
		editor.onIsCleanChanged.connect(function() {
			isCleanChanged(editor.isClean, document.documentId);
		});
	}

	function getEditor(documentId) {
		for (var i = 0; i < editorListModel.count; i++)
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
		for (var i = 0; i < editorListModel.count; i++)
			if (editorListModel.get(i).documentId === currentDocumentId)
				return editorListModel.get(i).isContract;
		return false;
	}

	function getBreakpoints() {
		var bpMap = {};
		for (var i = 0; i < editorListModel.count; i++)  {
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

	Component.onCompleted: projectModel.codeEditor = codeEditorView;

	Connections {
		target: projectModel
		onDocumentOpened: {
			openDocument(document);
		}

		onProjectSaving: {
			for (var i = 0; i < editorListModel.count; i++)
			{
				var doc = editorListModel.get(i);
				var editor = editors.itemAt(i).item;
				if (editor)
					fileIo.writeFile(doc.path, editor.getText());
			}
		}

		onProjectSaved: {
			if (projectModel.appIsClosing || projectModel.projectIsClosing)
				return;
			for (var i = 0; i < editorListModel.count; i++)
			{
				var doc = editorListModel.get(i);
				resetEditStatus(doc.documentId);
			}
		}

		onProjectClosed: {
			for (var i = 0; i < editorListModel.count; i++)
				editors.itemAt(i).visible = false;
			editorListModel.clear();
			currentDocumentId = "";
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

	MessageDialog
	{
		id: messageDialog
		title: qsTr("File Changed")
		text: qsTr("This file has been changed outside of the editor. Do you want to reload it?")
		standardButtons: StandardButton.Yes | StandardButton.No
		property variant item
		property variant doc
		onYes: {
			doLoadDocument(item, doc);
			resetEditStatus(doc.documentId);
		}
	}

	Repeater {
		id: editors
		model: editorListModel
		delegate: Loader {
			id: loader
			active: false
			asynchronous: true
			anchors.fill:  parent
			source: "CodeEditor.qml"
			visible: (index >= 0 && index < editorListModel.count && currentDocumentId === editorListModel.get(index).documentId)
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
				}
			}
			Component.onCompleted: {
				loadIfNotLoaded()
			}
			onLoaded: {
				doLoadDocument(loader.item, editorListModel.get(index))
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
}
