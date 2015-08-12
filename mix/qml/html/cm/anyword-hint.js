(function() {
  "use strict";

  var WORD = /[\w$]+/, RANGE = 500;

  CodeMirror.registerHelper("hint", "anyword", function(editor, options) {
	var word = options && options.word || WORD;
	var range = options && options.range || RANGE;
	var cur = editor.getCursor(), curLine = editor.getLine(cur.line);
	var start = cur.ch, end = start;
	while (end < curLine.length && word.test(curLine.charAt(end))) ++end;
	while (start && word.test(curLine.charAt(start - 1))) --start;
	var curWord = start != end && curLine.slice(start, end);

	var list = [], seen = {};

	if (editor.getMode().name === "solidity")
	{
		list = addSolToken(curWord, list, seen, solCurrency(), solCurrency);
		list = addSolToken(curWord, list, seen, solKeywords(), solKeywords);
		list = addSolToken(curWord, list, seen, solStdContract(), solStdContract);
		list = addSolToken(curWord, list, seen, solTime(), solTime);
		list = addSolToken(curWord, list, seen, solTypes(), solTypes);
		list = addSolToken(curWord, list, seen, solMisc(), solMisc);
	}

	//TODO: tokenize properly
	if (curLine.slice(start - 6, start) === "block.")
		list = addSolToken(curWord, list, seen, solBlock(), solBlock);
	else if (curLine.slice(start - 4, start) === "msg.")
		list = addSolToken(curWord, list, seen, solMsg(), solMsg);
	else if (curLine.slice(start - 3, start) === "tx.")
		list = addSolToken(curWord, list, seen, solTx(), solTx);


	var previousWord = "";
	var re = new RegExp(word.source, "g");
	for (var dir = -1; dir <= 1; dir += 2) {
	  var line = cur.line, endLine = Math.min(Math.max(line + dir * range, editor.firstLine()), editor.lastLine()) + dir;
	  for (; line != endLine; line += dir) {
		var text = editor.getLine(line), m;
		while (m = re.exec(text)) {
		  if (line == cur.line && m[0] === curWord) continue;
		  if ((!curWord || m[0].lastIndexOf(curWord, 0) === 0) && !Object.prototype.hasOwnProperty.call(seen, m[0])) {
			seen[m[0]] = true;
			var w = { text: m[0] };
			checkDeclaration(previousWord, "Contract", w);
			checkDeclaration(previousWord, "Function", w);
			list.push(w);
		  }
		  previousWord = m[0];
		}
	  }
	}
	return {list: list, from: CodeMirror.Pos(cur.line, start), to: CodeMirror.Pos(cur.line, end)};
  });
})();


function addSolToken(curWord, list, seen, tokens, type)
{
	var keywordsTypeName = keywordsName();
	for (var key in tokens)
	{
		seen[key] = true;
		if (curWord === false || key.indexOf(curWord, 0) === 0)
		{
			var token = { text: key };
			token.render = function(elt, data, cur)
			{
				render(elt, data, cur, type.name.toLowerCase(), keywordsTypeName[type.name.toLowerCase()]);
			}
			list.push(token);
		}
	}
	return list;
}

function render(elt, data, cur, csstype, type)
{
	var container = document.createElement("div");
	var word = document.createElement("div");
	word.className = csstype + " solToken";
	word.appendChild(document.createTextNode(cur.displayText || cur.text));
	var typeDiv = document.createElement("type");
	typeDiv.appendChild(document.createTextNode(type));
	typeDiv.className = "solTokenType";
	container.appendChild(word);
	container.appendChild(typeDiv);
	elt.appendChild(container);
}

function checkDeclaration(previousToken, target, current)
{
	if (previousToken.toLowerCase() === target.toLowerCase())
	{
		current.render = function(elt, data, cur)
		{
			render(elt, data, cur, "sol" + target, target);
		}
	}
}
