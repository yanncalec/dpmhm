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
    tags: 'ams',
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true,
    packages: {'[+]': ['base','ams','amsmath','newcommand','configmacros','noerrors']},
    // https://docs.mathjax.org/en/latest/input/tex/extensions/configmacros.html#tex-configmacros-options
    macros : {
      R: "\\mathbb{R}",
      N: "\\mathbb{N}",
      Z: "\\mathbb{Z}",
      Q: "\\mathbb{Q}",
      C: "\\mathrm{C}",
      loss: '\\mathcal{L}',
      dd: '\\mathrm{d}',
      indic: '\\mathbbm{1}',

      RUL:'\\operatorname{RUL}',

      ELBO:'\\operatorname{ELBO}',
      MMD: '\\operatorname{MMD}',

      opKL:'\\operatorname{\\mathbb{D}_{KL}}',
      // opprob:'\\operatorname{\\mathrm{p}}',
      opProb:'\\operatorname{\\mathbb{P}}',
      Exp:'\\operatorname*{\\mathbb{E}}',
      Var:'\\operatorname*{\\mathbb{V}}',
      rank: '\\operatorname{rank}',
      opNormal: '\\operatorname{\\mathcal{N}}',

      argmax: '\\operatornamewithlimits{arg\,max}',
      argmin: '\\operatornamewithlimits{arg\,min}',
      id: '\\operatorname{id}',
      sign: '\\operatorname{sign}',
      // supp: '\\operatorname{supp}',

      integral: ['\\int_{#1}^{#2}{#3}\\dd{#4}', 4],
      integrald: ['\\int_{#1}{#2}\\dd{#3}', 3],
      integralR: ['\\int_{\\R}{#1}\\dd{#2}', 2],

      // bold: ["{\\bf #1}", 1],
      braced: ['\\left\\{ #1 \\right\\}', 1],
      bracket: ['\\left\[ #1 \\right\]', 1],
      paren: ['\\left( #1 \\right)', 1],
      abs: ['\\left| #1 \\right|', 1],
      norm: ['\\left\\| #1 \\right\\|', 1],
      KL: ['\\opKL\\paren{#1 || #2}', 2],
      Normal: ['\\opNormal\\paren{#1, #2}', 2],
      // Exp: ['\\opExp\\bracket{#1}', 1],
      // Var: ['\\opVar\\bracket{#1}', 1],
      // dkl: ['\\left(#1 || #2\\right)', 2],
      bold: ['\\boldsymbol{#1}',1] ,     // this macro has one parameter
      // ddx: ['\\frac{d#2}{d#1}', 2, 'x'], // this macro has an optional parameter that defaults to 'x'
      // abc: ['(#1)', 1, [null, '\\cba']]  // equivalent to \def\abc#1\cba{(#1)}
    },
    environments: {
      // Doesn't work:
      // braced: ["\\left\\{", "\\right\\}"],
      // paren: ["\\left(", "\\right)"],
      // bracket: ["\\left[", "\\right]"],
      // abs: ["\\left|", "\\right|"],
      // norm: ["\\left\\|", "\\right\\|"],
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
