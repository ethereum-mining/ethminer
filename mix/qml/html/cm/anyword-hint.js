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
	var re = new RegExp(word.source, "g");
	for (var dir = -1; dir <= 1; dir += 2) {
	  var line = cur.line, endLine = Math.min(Math.max(line + dir * range, editor.firstLine()), editor.lastLine()) + dir;
	  for (; line != endLine; line += dir) {
		var text = editor.getLine(line), m;
		while (m = re.exec(text)) {
		  if (line == cur.line && m[0] === curWord) continue;
		  if ((!curWord || m[0].lastIndexOf(curWord, 0) === 0) && !Object.prototype.hasOwnProperty.call(seen, m[0])) {
			seen[m[0]] = true;
			list.push({ text: m[0] });
		  }
		}
	  }
	}

	if (editor.getMode().name === "solidity")
	{
		list = addSolToken(curWord, list, solCurrency(), solCurrency);
		list = addSolToken(curWord, list, solKeywords(), solKeywords);
		list = addSolToken(curWord, list, solStdContract(), solStdContract);
		list = addSolToken(curWord, list, solTime(), solTime);
		list = addSolToken(curWord, list, solTypes(), solTypes);
		list = addSolToken(curWord, list, solMisc(), solMisc);
	}

	return {list: list, from: CodeMirror.Pos(cur.line, start), to: CodeMirror.Pos(cur.line, end)};
  });
})();


function addSolToken(curWord, list, tokens, type)
{
	for (var key in tokens)
	{
		if (curWord === false || key.indexOf(curWord, 0) === 0)
		{
			var token = { text: key };
			token.render = function(elt, data, cur)
			{
				elt.className = elt.className + " " + type.name.toLowerCase();
				elt.appendChild(document.createTextNode(cur.displayText || cur.text));
			}
			list.push(token);
		}
	}
	return list;
}
