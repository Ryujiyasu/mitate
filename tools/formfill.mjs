// Oxi（wasm）で帳票の .docx を扱う。差し込みは oxi-forms の lib/oxi-fill.js をそのまま使う。
//
//   node tools/formfill.mjs layout <template.docx> <outdir>
//       outdir/boxes.json  各欄の名前・種類・位置（docx_field_boxes。単位は pt、左上原点）
//       outdir/blank.pdf   空欄の様式（{{…}} を消したもの = 印刷して配る紙）
//   node tools/formfill.mjs fill <template.docx> <values.json> <out.docx> [out.pdf]
//       values.json の値を差し込んだ .docx（と PDF）
//
// 環境変数 OXI_FORMS（既定 ~/oxi-forms）、OXI_WASM（既定 tools/oxi-wasm）
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { resolve, dirname } from 'node:path';
import { homedir } from 'node:os';
import { pathToFileURL, fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const wasmDir = process.env.OXI_WASM || resolve(here, 'oxi-wasm');
const formsDir = process.env.OXI_FORMS || resolve(homedir(), 'oxi-forms');
const api = await import(pathToFileURL(resolve(wasmDir, 'oxidocs_wasm.js')).href);
await api.default({ module_or_path: readFileSync(resolve(wasmDir, 'oxidocs_wasm_bg.wasm')) });
const { fillDocx, baseName } = await import(pathToFileURL(resolve(formsDir, 'lib/oxi-fill.js')).href);

const [cmd, tplPath, ...rest] = process.argv.slice(2);
const tpl = readFileSync(tplPath);
if (cmd === 'layout') {
  const [out] = rest;
  mkdirSync(out, { recursive: true });
  const boxes = JSON.parse(api.docx_field_boxes(tpl));
  writeFileSync(resolve(out, 'boxes.json'), JSON.stringify(boxes, null, 1));
  const blank = Object.fromEntries(boxes.fields.map((f) => [baseName(f.name), '']));
  writeFileSync(resolve(out, 'blank.pdf'), api.docx_to_pdf(fillDocx(api, tpl, blank).docx));
  console.log(JSON.stringify({ fields: boxes.fields.map((f) => f.name) }));
} else if (cmd === 'fill') {
  const [valuesPath, outDocx, outPdf] = rest;
  const values = JSON.parse(readFileSync(valuesPath, 'utf8'));
  const { docx, missing } = fillDocx(api, tpl, values);
  writeFileSync(outDocx, docx);
  if (outPdf) writeFileSync(outPdf, api.docx_to_pdf(docx));
  console.log(JSON.stringify({ missing: missing || [] }));
} else {
  console.error('usage: layout <tpl> <outdir> | fill <tpl> <values.json> <out.docx> [out.pdf]');
  process.exit(2);
}
