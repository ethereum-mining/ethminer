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
		  if ((!curWord || m[0].lastIndexOf(curWord, 0) == 0) && !Object.prototype.hasOwnProperty.call(seen, m[0])) {
			seen[m[0]] = true;
			list.push(m[0]);
		  }
		}
	  }
	}

	for (var key in solidityKeywords())
	{
		if (curWord === false || key.indexOf(curWord, 0) === 0)
			list.push(key);
	}

	return {list: list, from: CodeMirror.Pos(cur.line, start), to: CodeMirror.Pos(cur.line, end)};
  });
})();


solidityKeywords = function(list)
{
	var keywords = { "address":true, "indexed":true, "event":true, "delete":true, "break":true, "case":true, "constant":true, "continue":true, "contract":true, "default":true,
		  "do":true, "else":true, "is":true, "for":true, "function":true, "if":true, "import":true, "mapping":true, "new":true,
		  "public":true, "private":true, "return":true, "returns":true, "struct":true, "switch":true, "var":true, "while":true,
		  "int":true, "uint":true, "hash":true, "bool":true, "string":true, "string0":true, "text":true, "real":true,
		  "ureal":true,
		  "owned":true,
		  "onlyowner":true,
		  "named":true,
		  "mortal":true,
		  "coin":true
	  };

	for (var i = 1; i <= 32; i++) {
		  keywords["int" + i * 8] = true;
		  keywords["uint" + i * 8] = true;
		  keywords["hash" + i * 8] = true;
		  keywords["string" + i] = true;
	};

	keywords["true"] = true;
	keywords["false"] = true;
	keywords["null"] = true;
	keywords["Config"] = true;
	keywords["NameReg"] = true;
	keywords["CoinReg"] = true;

	return keywords;
}


