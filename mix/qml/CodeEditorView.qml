import QtQuick 2.0
import QtQuick.Window 2.0
import QtQuick.Layouts 1.0
import QtQuick.Controls 1.0

Item {
	id: codeEditorView
	property string currentDocumentId: ""
	signal documentEdit(string documentId)
	signal breakpointsChanged(string documentId)

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
	}

	function getEditor(documentId) {
		for (var i = 0; i < editorListModel.count; i++)
			if (editorListModel.get(i).documentId === documentId)
				return editors.itemAt(i).item;
		return null;
	}

	function highlightExecution(documentId, location) {
		var editor = getEditor(documentId);
		if (editor)
			editor.highlightExecution(location);
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

	Component.onCompleted: projectModel.codeEditor = codeEditorView;

	Connections {
		target: projectModel
		onDocumentOpened: {
			openDocument(document);
		}
		onProjectSaving: {
			for (var i = 0; i < editorListModel.count; i++)
				fileIo.writeFile(editorListModel.get(i).path, editors.itemAt(i).item.getText());
		}
		onProjectClosed: {
			for (var i = 0; i < editorListModel.count; i++)	{
				editors.itemAt(i).visible = false;
			}
			editorListModel.clear();
			currentDocumentId = "";
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
			onVisibleChanged: {
				loadIfNotLoaded()
				if (visible && item)
					loader.item.setFocus();
			}
			Component.onCompleted: {
				loadIfNotLoaded()
			}
			onLoaded: {
				doLoadDocument(loader.item, editorListModel.get(index))
			}

			function loadIfNotLoaded () {
				if(visible && !active) {
					active = true;
				}
			}
		}
	}
	ListModel {
		id: editorListModel
	}
}
