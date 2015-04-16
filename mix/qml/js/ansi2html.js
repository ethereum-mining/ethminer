var ansi2html = (function(){

  function ansi2htmlI(str) {
    // this lib do not support \e
    str = str.replace(/e\[/g, '[');
    // nor line breaks
    str = '<div>' + str.replace(/\n/g, '</div><div>') + '</div>';
    var props = {}
      , open = false

    var stylemap =
      { bold: "font-weight"
      , underline: "text-decoration"
      , color: "color"
      , background: "background"
      }

    function style() {
      var key, val, style = []
      for (var key in props) {
        val = props[key]
        if (!val) continue
        if (val == true) {
          style.push(stylemap[key] + ':' + key)
        } else {
          style.push(stylemap[key] + ':' + val)
        }
      }
      return style.join(';')
    }


    function tag(code) {
      var i
        , tag = ''
        , n = ansi2htmlI.table[code]

      if (open) tag += '</span>'
      open = false

      if (n) {
        for (i in n) props[i] = n[i]
        tag += '<span style="' + style() + '">'
        open = true
      } else {
        props = {}
      }

      return tag
    }

    return str.replace(/\[(\d+;)?(\d+)*m/g, function(match, b1, b2) {
      var i, code, res = ''
      if (b2 == '' || b2 == null) b2 = '0'
      for (i = 1; i < arguments.length - 2; i++) {
        if (!arguments[i]) continue
        code = parseInt(arguments[i])
        res += tag(code)
      }
      return res
    }) + tag()
  }

  /* not implemented:
   *   italic
   *   blink
   *   invert
   *   strikethrough
   */
  ansi2htmlI.table =
  { 0: null
  , 1: { bold: true }
  , 3: { italic: true }
  , 4: { underline: true }
  , 5: { blink: true }
  , 6: { blink: true }
  , 7: { invert: true }
  , 9: { strikethrough: true }
  , 23: { italic: false }
  , 24: { underline: false }
  , 25: { blink: false }
  , 27: { invert: false }
  , 29: { strikethrough: false }
  , 30: { color: 'black' }
  , 31: { color: 'red' }
  , 32: { color: 'green' }
  , 33: { color: 'yellow' }
  , 34: { color: 'blue' }
  , 35: { color: 'magenta' }
  , 36: { color: 'cyan' }
  , 37: { color: 'white' }
  , 39: { color: null }
  , 40: { background: 'black' }
  , 41: { background: 'red' }
  , 42: { background: 'green' }
  , 43: { background: 'yellow' }
  , 44: { background: 'blue' }
  , 45: { background: 'magenta' }
  , 46: { background: 'cyan' }
  , 47: { background: 'white' }
  , 49: { background: null }
  }

  return ansi2htmlI;
})();

