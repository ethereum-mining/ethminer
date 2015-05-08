var editor = CodeMirror(document.body, {
							lineNumbers: true,
							//styleActiveLine: true,
							matchBrackets: true,
							autofocus: true,
							gutters: ["CodeMirror-linenumbers", "breakpoints"],
							autoCloseBrackets: true,
							styleSelectedText: true
						});
var ternServer;

editor.setOption("theme", "inkpot");
editor.setOption("indentUnit", 4);
editor.setOption("indentWithTabs", true);
editor.setOption("fullScreen", true);

editor.changeRegistered = false;
editor.breakpointsChangeRegistered = false;

editor.on("change", function(eMirror, object) {
	editor.changeRegistered = true;
});

var mac = /Mac/.test(navigator.platform);
var extraKeys = {};
if (mac === true) {
	extraKeys["Cmd-V"] = function(cm) { cm.replaceSelection(clipboard); };
	extraKeys["Cmd-X"] = function(cm) { window.document.execCommand("cut"); };
	extraKeys["Cmd-C"] = function(cm) { window.document.execCommand("copy"); };
}

makeMarker = function() {
	var marker = document.createElement("div");
	marker.style.color = "#822";
	marker.innerHTML = "●";
	return marker;
};

toggleBreakpointLine = function(n) {
	var info = editor.lineInfo(n);
	editor.setGutterMarker(n, "breakpoints", info.gutterMarkers ? null : makeMarker());
	editor.breakpointsChangeRegistered = true;
}

editor.on("gutterClick", function(cm, n) {
	toggleBreakpointLine(n);
});

toggleBreakpoint = function() {
	var line = editor.getCursor().line;
	toggleBreakpointLine(line);
}

getTextChanged = function() {
	return editor.changeRegistered;
};

getText = function() {
	editor.changeRegistered = false;
	return editor.getValue();
};

getBreakpointsChanged = function() {
	return editor.changeRegistered || editor.breakpointsChangeRegistered;   //TODO: track new lines
};

getBreakpoints = function() {
	var locations = [];
	editor.breakpointsChangeRegistered = false;
	var doc = editor.doc;
	doc.iter(function(line) {
		if (line.gutterMarkers && line.gutterMarkers["breakpoints"]) {
			var l = doc.getLineNumber(line);
			locations.push({
							   start: editor.indexFromPos({ line: l, ch: 0}),
							   end: editor.indexFromPos({ line: l + 1, ch: 0})
						   });;
		}
	});
	return locations;
};

setTextBase64 = function(text) {
	editor.setValue(window.atob(text));
	editor.getDoc().clearHistory();
	editor.focus();
};

setText = function(text) {
	editor.setValue(text);
};

setMode = function(mode) {
	this.editor.setOption("mode", mode);

	if (mode === "javascript")
	{
		ternServer = new CodeMirror.TernServer({defs: [ ecma5Spec() ]});
		extraKeys["Ctrl-Space"] = function(cm) { ternServer.complete(cm); };
		extraKeys["Ctrl-I"] = function(cm) { ternServer.showType(cm); };
		extraKeys["Ctrl-O"] = function(cm) { ternServer.showDocs(cm); };
		extraKeys["Alt-."] = function(cm) { ternServer.jumpToDef(cm); };
		extraKeys["Alt-,"] = function(cm) { ternServer.jumpBack(cm); };
		extraKeys["Ctrl-Q"] = function(cm) { ternServer.rename(cm); };
		extraKeys["Ctrl-."] = function(cm) { ternServer.selectName(cm); };
		extraKeys["'.'"] = function(cm) { setTimeout(function() { ternServer.complete(cm); }, 100); throw CodeMirror.Pass; };
		editor.on("cursorActivity", function(cm) { ternServer.updateArgHints(cm); });
	}
	else if (mode === "solidity")
	{
		CodeMirror.commands.autocomplete = function(cm) {
			CodeMirror.showHint(cm, CodeMirror.hint.anyword);
		}
		extraKeys["Ctrl-Space"] = "autocomplete";
	}
	editor.setOption("extraKeys", extraKeys);
};

setClipboardBase64 = function(text) {
	clipboard = window.atob(text);
};

var executionMark;
highlightExecution = function(start, end) {
	if (executionMark)
		executionMark.clear();
	if (debugWarning)
		debugWarning.clear();
	if (start > 0 && end > start) {
		executionMark = editor.markText(editor.posFromIndex(start), editor.posFromIndex(end), { className: "CodeMirror-exechighlight" });
		editor.scrollIntoView(editor.posFromIndex(start));
	}
}

var changeId;
changeGeneration = function()
{
	changeId = editor.changeGeneration(true);
}

isClean = function()
{
	return editor.isClean(changeId);
}

var debugWarning = null;
showWarning = function(content)
{
	if (executionMark)
		executionMark.clear();
	if (debugWarning)
		debugWarning.clear();
	var node = document.createElement("div");
	node.id = "annotation"
	node.innerHTML = content;
	node.className = "CodeMirror-errorannotation-context";
	debugWarning = editor.addLineWidget(0, node, { coverGutter: false, above: true });
}

var annotation = null;
var compilationCompleteBool = true;
compilationError = function(line, column, content)
{
	compilationCompleteBool = false;
	window.setTimeout(function(){
		if (compilationCompleteBool)
			return;
		line = parseInt(line);
		column = parseInt(column);
		if (line > 0)
			line = line - 1;
		if (column > 0)
			column = column - 1;

		if (annotation == null)
			annotation = new ErrorAnnotation(editor, line, column, content);
		else if (annotation.line !== line || annotation.column !== column || annotation.content !== content)
		{
			annotation.destroy();
			annotation = new ErrorAnnotation(editor, line, column, content);
		}
	}, 500)
}

compilationComplete = function()
{
	if (annotation !== null)
	{
		annotation.destroy();
		annotation = null;
	}
	compilationCompleteBool = true;
}

goToCompilationError = function()
{
	editor.setCursor(annotation.line, annotation.column)
}

setFontSize = function(size)
{
	editor.getWrapperElement().style["font-size"] = size + "px";
	editor.refresh();
}

editor.setOption("extraKeys", extraKeys);

