// 用 DOM 桩把规划文档渲染一遍，检查「标签闭合」之外的事。
const fs = require('fs');
const { JSDOM } = require('jsdom');

const path = 'C:\\Users\\宿心\\Desktop\\ICLR2027投稿规划.html';
const dom = new JSDOM(fs.readFileSync(path, 'utf8'));
const doc = dom.window.document;

let problems = 0;
const bad = (m) => { problems++; console.log('  ✗ ' + m); };

// 一、章节编号连续
const heads = [...doc.querySelectorAll('.secnum')].map(e => +e.textContent.trim());
console.log('章节：' + heads.join(', '));
heads.forEach((n, i) => { if (n !== i + 1) bad('章节号不连续：第 ' + (i + 1) + ' 个是 ' + n); });

// 二、表格列数（rowspan/colspan 计入）
let t = 0;
for (const tbl of doc.querySelectorAll('table')) {
  t++;
  const hr = tbl.querySelector('thead tr');
  if (!hr) continue;
  const ncol = [...hr.children].reduce((n, c) => n + (+(c.getAttribute('colspan') || 1)), 0);
  let r = 0;
  for (const row of tbl.querySelectorAll('tbody tr')) {
    r++;
    const n = [...row.children].reduce((s, c) => s + (+(c.getAttribute('colspan') || 1)), 0);
    if (n !== ncol) bad('表 ' + t + ' 第 ' + r + ' 行 ' + n + ' 列，表头 ' + ncol + ' 列');
  }
}
console.log('表格：' + t + ' 张');

// 三、class 都有样式定义
const css = [...doc.querySelectorAll('style')].map(s => s.textContent).join('\n');
const used = new Set();
for (const el of doc.querySelectorAll('[class]')) {
  el.getAttribute('class').split(/\s+/).filter(Boolean).forEach(c => used.add(c));
}
const missing = [...used].filter(c => !new RegExp('\\.' + c + '\\b').test(css));
if (missing.length) bad('无样式定义的 class：' + missing.join(', '));
console.log('class：' + used.size + ' 个');

// 四、关键结构存在
for (const sel of ['.masthead', '.verdict', '.countdown', '.gantt']) {
  if (!doc.querySelector(sel)) bad('缺少结构 ' + sel);
}
console.log('倒计时卡片：' + doc.querySelectorAll('.cd').length + ' 个');
console.log('实验条目：' + doc.querySelectorAll('.lane').length + ' 条');

// 五、实体已解析、无裸露的 markdown
const text = doc.body.textContent;
for (const ent of ['&yen;', '&ndash;', '&mdash;', '&plusmn;', '&times;', '&ge;']) {
  if (text.includes(ent)) bad('实体未解析：' + ent);
}
if (/\*\*/.test(text)) bad('残留 markdown 星号');

// 六、日期一致性 —— 全文出现的截止日期必须一致
const dl = [...text.matchAll(/9\/(\d\d)|2026-09-(\d\d)/g)].map(m => m[1] || m[2]);
const uniq = [...new Set(dl)].sort();
console.log('文中出现的九月日期：' + uniq.join(', '));
if (!uniq.includes('18') || !uniq.includes('25')) bad('缺少 9/18 或 9/25 两个关键截止日');

console.log(problems === 0 ? '\n通过：未发现问题' : '\n发现 ' + problems + ' 处问题');
process.exit(problems === 0 ? 0 : 1);
