// Copied from
// https://squidfunk.github.io/mkdocs-material/reference/mathjax/

// For configuring MathJax, see:
// https://docs.mathjax.org/en/latest/options/index.html

window.MathJax = {
  loader: {
    // For MathML and AsciiMath, see
    // https://docs.mathjax.org/en/latest/basic/mathematics.html
    load: ['input/tex-base','output/chtml','[tex]/ams','[tex]/newcommand','[tex]/configmacros','[tex]/noerrors']
  },

  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true,
    packages: {'[+]': ['base','ams','newcommand','configmacros','noerrors']},
    // https://docs.mathjax.org/en/latest/input/tex/extensions/configmacros.html#tex-configmacros-options
    macros : {
      R: "\\mathbb{R}",
      N: "\\mathbb{N}",
      Z: "\\mathbb{Z}",
      Q: "\\mathbb{Q}",
      C: "\\mathrm{C}",
      rank: '\\operatorname{rank}',
      bold: ["{\\bf #1}", 1],
      // abc: ['(#1)', 1, [null, '\\cba']]  // equivalent to \def\abc#1\cba{(#1)}
    },
    environments: {
      braced: ["\\left\\{", "\\right\\}"]
    }
  },

  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex"
  }
};

document$.subscribe(() => {
  MathJax.typesetPromise()
})
