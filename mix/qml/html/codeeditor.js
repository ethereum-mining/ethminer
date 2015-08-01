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

var annotations = [];
var compilationCompleteBool = true;
compilationError = function(currentSourceName, location, error, secondaryErrors)
{
	compilationCompleteBool = false;
	if (compilationCompleteBool)
		return;
	clearAnnotations();
	location = JSON.parse(location);
	if (location.source === currentSourceName)
		ensureAnnotation(location, error, "first");
	var lineError = location.start.line + 1;
	var errorOrigin = "Source " + location.contractName + " line " + lineError;
	secondaryErrors = JSON.parse(secondaryErrors);
	for(var i in secondaryErrors)
	{
		if (secondaryErrors[i].source === currentSourceName)
			ensureAnnotation(secondaryErrors[i], errorOrigin, "second");
	}
}

ensureAnnotation = function(location, error, type)
{
	annotations.push({ "type": type, "annotation": new ErrorAnnotation(editor, location, error)});
}

clearAnnotations = function()
{
	for (var k in annotations)
		annotations[k].annotation.destroy();
	annotations.length = 0;
}

compilationComplete = function()
{
	clearAnnotations();
	compilationCompleteBool = true;
}

goToCompilationError = function()
{
	if (annotations.length > 0)
		editor.setCursor(annotations[0].annotation.location.start.line, annotations[0].annotation.location.start.column)
}

setFontSize = function(size)
{
	editor.getWrapperElement().style["font-size"] = size + "px";
	editor.refresh();
}

makeGasCostMarker = function(value) {
	var marker = document.createElement("div");
	marker.innerHTML = value;
	marker.className = "CodeMirror-gasCost";
	return marker;
};

var gasCosts = null;
setGasCosts = function(_gasCosts)
{
	gasCosts = JSON.parse(_gasCosts);
	if (showingGasEstimation)
	{
		displayGasEstimation(false);
		displayGasEstimation(true);
	}
}

var showingGasEstimation = false;
var gasMarkText = [];
var gasMarkRef = {};
displayGasEstimation = function(show)
{
	show = JSON.parse(show);
	showingGasEstimation = show;
	if (show)
	{
		var maxGas = 20000;
		var step = colorGradient.length / maxGas; // 20000 max gas
		clearGasMark();
		gasMarkText = [];
		gasMarkRef = {};
		for (var i in gasCosts)
		{
			if (gasCosts[i].gas !== "0")
			{
				var color;
				var colorIndex = Math.round(step * gasCosts[i].gas);
				if (gasCosts[i].isInfinite || colorIndex >= colorGradient.length)
					color = colorGradient[colorGradient.length - 1];
				else
					color = colorGradient[colorIndex];
				var className = "CodeMirror-gasCosts" + i;
				var line = editor.posFromIndex(gasCosts[i].start);
				var endChar;
				if (gasCosts[i].codeBlockType === "statement" || gasCosts[i].codeBlockType === "")
				{
					endChar = editor.posFromIndex(gasCosts[i].end);
					gasMarkText.push({ line: line, markText: editor.markText(line, endChar, { inclusiveLeft: true, inclusiveRight: true, handleMouseEvents: true, className: className, css: "background-color:" + color })});
				}
				else if (gasCosts[i].codeBlockType === "function" || gasCosts[i].codeBlockType === "constructor")
				{
					var l = editor.getLine(line.line);
					endChar = { line: line.line, ch: line.ch + l.length };
					var marker = document.createElement("div");
					marker.innerHTML = " max execution cost: " + gasCosts[i].gas + " gas";
					marker.className = "CodeMirror-gasCost";
					editor.addWidget(endChar, marker, false, "over");
					gasMarkText.push({ line: line.line, widget: marker });
				}
				gasMarkRef[className] = { line: line.line, value: gasCosts[i] };
			}
		}
		CodeMirror.on(editor.getWrapperElement(), "mouseover", listenMouseOver);
	}
	else
	{
		CodeMirror.off(editor.getWrapperElement(), "mouseover", listenMouseOver);
		clearGasMark();
		if (gasAnnotation)
		{
			gasAnnotation.clear();
			gasAnnotation = null;
		}
	}
}

function clearGasMark()
{
	if (gasMarkText)
		for (var k in gasMarkText)
		{
			if (gasMarkText[k] && gasMarkText[k].markText)
				gasMarkText[k].markText.clear();
			if (gasMarkText[k] && gasMarkText[k].widget)
				gasMarkText[k].widget.remove();
		}
}

var gasAnnotation;
function listenMouseOver(e)
{
	var node = e.target || e.srcElement;
	if (node)
	{
		if (node.className && node.className.indexOf("CodeMirror-gasCosts") !== -1)
		{
			if (gasAnnotation)
				gasAnnotation.clear();
			var cl = getGasCostClass(node);
			var gasTitle = gasMarkRef[cl].value.isInfinite ? "infinite" : gasMarkRef[cl].value.gas;
			gasTitle = " execution cost: " + gasTitle + " gas";
			gasAnnotation = editor.addLineWidget(gasMarkRef[cl].line + 1, makeGasCostMarker(gasTitle), { coverGutter: false, above: true });
		}
		else if (gasAnnotation)
		{
			gasAnnotation.clear();
			gasAnnotation = null;
		}
	}
}

function getGasCostClass(node)
{
	var classes = node.className.split(" ");
	for (var k in classes)
	{
		if (classes[k].indexOf("CodeMirror-gasCosts") !== -1)
			return classes[k];
	}
	return "";
}

// blue => red ["#1515ED", "#1714EA", "#1914E8", "#1B14E6", "#1D14E4", "#1F14E2", "#2214E0", "#2414DE", "#2614DC", "#2813DA", "#2A13D8", "#2D13D6", "#2F13D4", "#3113D2", "#3313D0", "#3513CE", "#3713CC", "#3A12CA", "#3C12C8", "#3E12C6", "#4012C4", "#4212C2", "#4512C0", "#4712BE", "#4912BC", "#4B11BA", "#4D11B8", "#4F11B6", "#5211B4", "#5411B2", "#5611B0", "#5811AE", "#5A11AC", "#5D11AA", "#5F10A7", "#6110A5", "#6310A3", "#6510A1", "#67109F", "#6A109D", "#6C109B", "#6E1099", "#700F97", "#720F95", "#750F93", "#770F91", "#790F8F", "#7B0F8D", "#7D0F8B", "#7F0F89", "#820E87", "#840E85", "#860E83", "#880E81", "#8A0E7F", "#8D0E7D", "#8F0E7B", "#910E79", "#930D77", "#950D75", "#970D73", "#9A0D71", "#9C0D6F", "#9E0D6D", "#A00D6B", "#A20D69", "#A50D67", "#A70C64", "#A90C62", "#AB0C60", "#AD0C5E", "#AF0C5C", "#B20C5A", "#B40C58", "#B60C56", "#B80B54", "#BA0B52", "#BD0B50", "#BF0B4E", "#C10B4C", "#C30B4A", "#C50B48", "#C70B46", "#CA0A44", "#CC0A42", "#CE0A40", "#D00A3E", "#D20A3C", "#D50A3A", "#D70A38", "#D90A36", "#DB0934", "#DD0932", "#DF0930", "#E2092E", "#E4092C", "#E6092A", "#E80928", "#EA0926", "#ED0924"]
/* green => red */ var colorGradient = ["#429C27", "#439A26", "#449926", "#469726", "#479626", "#489525", "#4A9325", "#4B9225", "#4D9025", "#4E8F25", "#4F8E24", "#518C24", "#528B24", "#548924", "#558824", "#568723", "#588523", "#598423", "#5B8223", "#5C8122", "#5D8022", "#5F7E22", "#607D22", "#627B22", "#637A21", "#647921", "#667721", "#677621", "#697421", "#6A7320", "#6B7220", "#6D7020", "#6E6F20", "#706E20", "#716C1F", "#726B1F", "#74691F", "#75681F", "#76671E", "#78651E", "#79641E", "#7B621E", "#7C611E", "#7D601D", "#7F5E1D", "#805D1D", "#825B1D", "#835A1D", "#84591C", "#86571C", "#87561C", "#89541C", "#8A531B", "#8B521B", "#8D501B", "#8E4F1B", "#904D1B", "#914C1A", "#924B1A", "#94491A", "#95481A", "#97461A", "#984519", "#994419", "#9B4219", "#9C4119", "#9E4019", "#9F3E18", "#A03D18", "#A23B18", "#A33A18", "#A43917", "#A63717", "#A73617", "#A93417", "#AA3317", "#AB3216", "#AD3016", "#AE2F16", "#B02D16", "#B12C16", "#B22B15", "#B42915", "#B52815", "#B72615", "#B82514", "#B92414", "#BB2214", "#BC2114", "#BE1F14", "#BF1E13", "#C01D13", "#C21B13", "#C31A13", "#C51813", "#C61712", "#C71612", "#C91412", "#CA1312", "#CC1212"]

editor.setOption("extraKeys", extraKeys);

