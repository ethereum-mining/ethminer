function ErrorAnnotation(editor, line, column, content)
{
	this.opened = false;
	this.line = line;
	this.column = column;
	this.content = content.replace("Contract Error:", "");
	this.editor = editor;
	this.errorMark = null;
	this.lineWidget = null;
	this.init();
	this.open();
}

ErrorAnnotation.prototype.init = function()
{
	var separators = [';', ',', '\\\(', '\\\{',  '\\\}', '\\\)', ':'];
	var errorPart = editor.getLine(this.line).substring(this.column);
	var incrMark = this.column + errorPart.split(new RegExp(separators.join('|'), 'g'))[0].length;
	if (incrMark === this.column)
		incrMark = this.column + 1;
	this.errorMark = editor.markText({ line: this.line, ch: this.column }, { line: this.line, ch: incrMark }, { className: "CodeMirror-errorannotation", inclusiveRight: true });
}

ErrorAnnotation.prototype.open = function()
{
	if (this.line)
	{
		var node = document.createElement("div");
		node.id = "annotation"
		node.innerHTML = this.content;
		node.className = "CodeMirror-errorannotation-context";
		this.lineWidget = this.editor.addLineWidget(this.line, node, { coverGutter: false });
		this.opened = true;
	}
}

ErrorAnnotation.prototype.close = function()
{
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
