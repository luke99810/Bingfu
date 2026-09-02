// 用 DOM 桩把设计文档真渲染一遍，检查「元素存在」之外的事。
//
// ★ 标签闭合通过 ≠ 页面对。要查的是：
//   - 新加的小节有没有真的挂进文档结构（不是被吞进上一个 <p>）
//   - 表格的列数在 DOM 解析后是否一致（rowspan/colspan 算进去）
//   - 用到的 class 是否都在样式表里定义过
//   - 章节编号有没有重号或断号
const fs = require('fs');
const { JSDOM } = require('jsdom');

const html = fs.readFileSync('C:\\Users\\宿心\\Desktop\\兵符框架设计.html', 'utf8');
const dom = new JSDOM(html);
const doc = dom.window.document;

let problems = 0;
const bad = (m) => { problems++; console.log('  ✗ ' + m); };

// ── 一、章节编号 ─────────────────────────────────────
const heads = [...doc.querySelectorAll('h2, h3')].map(h => h.textContent.trim());
const numbered = heads.filter(t => /^\d+\.\d+\s/.test(t));
console.log('小节数：' + numbered.length);
const seen = new Set();
let prevMajor = null, prevMinor = 0;
for (const t of numbered) {
  const m = t.match(/^(\d+)\.(\d+)/);
  const key = m[1] + '.' + m[2];
  if (seen.has(key)) bad('章节号重复：' + t.slice(0, 34));
  seen.add(key);
  const [maj, min] = [+m[1], +m[2]];
  if (maj === prevMajor && min !== prevMinor + 1) {
    bad('章节号不连续：' + prevMajor + '.' + prevMinor + ' → ' + t.slice(0, 34));
  }
  prevMajor = maj; prevMinor = min;
}

// ── 二、表格列数（DOM 解析后，rowspan/colspan 已生效）──
let tblIdx = 0;
for (const tbl of doc.querySelectorAll('table')) {
  tblIdx++;
  const cap = (tbl.querySelector('caption') || {}).textContent || '(无标题)';
  const headRow = tbl.querySelector('thead tr');
  if (!headRow) continue;
  const ncol = [...headRow.children]
    .reduce((n, c) => n + (parseInt(c.getAttribute('colspan') || 1, 10)), 0);
  // 逐行累计 rowspan 占位
  const carry = {};
  let r = 0;
  for (const row of tbl.querySelectorAll('tbody tr')) {
    r++;
    let n = 0;
    for (const k in carry) { if (carry[k] > 0) { n++; carry[k]--; } }
    for (const c of row.children) {
      const cs = parseInt(c.getAttribute('colspan') || 1, 10);
      const rs = parseInt(c.getAttribute('rowspan') || 1, 10);
      n += cs;
      if (rs > 1) carry[r + '_' + n] = rs - 1;
    }
    if (n !== ncol) {
      bad('表 ' + tblIdx + '「' + cap.trim().slice(0, 26) + '」第 ' + r
          + ' 行 ' + n + ' 列，表头 ' + ncol + ' 列');
    }
  }
}
console.log('表格数：' + tblIdx);

// ── 三、class 是否都有样式 ───────────────────────────
const css = [...doc.querySelectorAll('style')].map(s => s.textContent).join('\n');
const used = new Set();
for (const el of doc.querySelectorAll('[class]')) {
  el.getAttribute('class').split(/\s+/).filter(Boolean).forEach(c => used.add(c));
}
const undefinedClasses = [...used].filter(c => !new RegExp('\\.' + c + '\\b').test(css));
if (undefinedClasses.length) bad('用了但样式表里没有的 class：' + undefinedClasses.join(', '));
console.log('用到的 class：' + used.size + ' 个');

// ── 四、新小节确实在文档结构里，不是被吞掉 ────────────
for (const want of ['12.5', '11.5', '11.6']) {
  const h = heads.find(t => t.startsWith(want + ' '));
  if (!h) bad('找不到小节 ' + want);
}
// 12.5 下面应当有 3 张表与若干 note
const h125 = [...doc.querySelectorAll('h3')].find(h => h.textContent.startsWith('12.5'));
if (h125) {
  let n = 0, notes = 0, el = h125.nextElementSibling;
  while (el && el.tagName !== 'H3') {
    n += el.querySelectorAll ? el.querySelectorAll('table').length : 0;
    if (el.className && el.className.includes('note')) notes++;
    el = el.nextElementSibling;
  }
  console.log('12.5 节内：表 ' + n + ' 张，note ' + notes + ' 个');
  // 12.5 有三张表：按任务的 token 对照、修复后 120 次效果、残余差距归因
  if (n < 3) bad('12.5 的表少了，应有 ≥3 张');
  if (notes < 3) bad('12.5 的 note 少了');
}

// ── 五、有没有裸露的实体或未转义的尖括号残留 ──────────
const text = doc.body.textContent;
for (const ent of ['&ndash;', '&mdash;', '&yen;', '&ge;', '&minus;', '&sim;', '&rarr;']) {
  if (text.includes(ent)) bad('实体没被解析，会显示成字面文本：' + ent);
}

console.log(problems === 0 ? '\n通过：没有发现问题' : '\n发现 ' + problems + ' 处问题');
process.exit(problems === 0 ? 0 : 1);
