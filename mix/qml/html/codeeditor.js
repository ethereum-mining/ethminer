
var editor = CodeMirror(document.body, {
							lineNumbers: true,
							styleActiveLine: true,
							matchBrackets: true,
							autofocus: true,

						});
editor.setOption("theme", "blackboard");
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


setText = function(text) {
	editor.setValue(text);
};
