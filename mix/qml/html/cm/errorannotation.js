function ErrorAnnotation(editor, location, content)
{
	this.location = location;
	this.opened = false;
	this.rawContent = content;
	this.content = content.replace("Contract Error:", "");
	this.editor = editor;
	this.errorMark = null;
	this.lineWidget = null;
	this.init();
	if (this.content)
		this.open();
}

ErrorAnnotation.prototype.init = function()
{
	this.errorMark = editor.markText({ line: this.location.start.line, ch: this.location.start.column }, { line: this.location.end.line, ch: this.location.end.column }, { className: "CodeMirror-errorannotation", inclusiveRight: true });
}

ErrorAnnotation.prototype.open = function()
{
	if (this.location.start.line)
	{
		var node = document.createElement("div");
		node.id = "annotation"
		node.innerHTML = this.content;
		node.className = "CodeMirror-errorannotation-context";
		this.lineWidget = this.editor.addLineWidget(this.location.start.line, node, { coverGutter: false });
		this.opened = true;
	}
}

ErrorAnnotation.prototype.close = function()
{
	if (this.lineWidget)
		this.lineWidget.clear();
	this.opened = false;
}

ErrorAnnotation.prototype.destroy = function()
{
	if (this.opened)
		this.close();
	if (this.errorMark)
		this.errorMark.clear();
}
