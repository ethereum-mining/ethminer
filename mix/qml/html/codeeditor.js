
var editor = CodeMirror(document.body, {
							lineNumbers: true,
							//styleActiveLine: true,
							matchBrackets: true,
							autofocus: true,
						});

editor.setOption("theme", "solarized dark");
editor.setOption("indentUnit", 4);
editor.setOption("indentWithTabs", true);
editor.setOption("fullScreen", true);

editor.changeRegistered = false;

editor.on("change", function(eMirror, object) {
	editor.changeRegistered = true;

});

getTextChanged = function() {
	return editor.changeRegistered;
};


getText = function() {
	editor.changeRegistered = false;
	return editor.getValue();
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
};
