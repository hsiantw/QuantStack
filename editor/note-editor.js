import {Editor} from '@tiptap/core';
import StarterKit from '@tiptap/starter-kit';
import {TableKit} from '@tiptap/extension-table';
import Highlight from '@tiptap/extension-highlight';
import TextAlign from '@tiptap/extension-text-align';
import {TextStyle, Color, FontSize, FontFamily} from '@tiptap/extension-text-style';
import TaskList from '@tiptap/extension-task-list';
import TaskItem from '@tiptap/extension-task-item';
import CharacterCount from '@tiptap/extension-character-count';
import Subscript from '@tiptap/extension-subscript';
import Superscript from '@tiptap/extension-superscript';
import DOMPurify from 'dompurify';

const fonts = ['Arial', 'Georgia', 'monospace'];
const safeURL = value => {
  try { const url = new URL(value); return ['http:', 'https:', 'mailto:'].includes(url.protocol) ? url.href : null; }
  catch { return null; }
};
const allowedTags = ['p','br','div','span','strong','b','em','i','u','s','strike','del','h1','h2','h3','ul','ol','li','blockquote','pre','code','hr','a','mark','sub','sup','table','thead','tbody','tfoot','tr','th','td','label','input'];
DOMPurify.addHook('afterSanitizeAttributes', node => {
  if (!(node instanceof Element)) return;
  if (node.hasAttribute('style')) {
    const original = node.style;
    const kept = [];
    for (const prop of ['color','background-color','text-align','font-size','font-family']) {
      const value = original.getPropertyValue(prop).trim();
      const valid = ['color','background-color'].includes(prop) ? /^(#[0-9a-f]{3,8}|rgba?\([\d.,%\s]+\)|[a-z]+)$/i.test(value) :
        prop === 'text-align' ? ['left','center','right','justify'].includes(value) :
        prop === 'font-size' ? /^(1[0-9]|2[0-9]|3[0-6])px$/.test(value) : fonts.includes(value.replaceAll('"','').replaceAll("'",''));
      if (valid) kept.push(`${prop}:${value}`);
    }
    node.removeAttribute('style');
    if (kept.length) node.setAttribute('style', kept.join(';'));
  }
  if (node.tagName === 'A') {
    const href = safeURL(node.getAttribute('href'));
    if (href) { node.setAttribute('href',href); node.setAttribute('rel','noopener noreferrer'); }
    else node.removeAttribute('href');
  }
  if (node.tagName === 'INPUT') { node.setAttribute('type','checkbox'); node.setAttribute('disabled',''); }
  for (const attr of ['colspan','rowspan']) if (node.hasAttribute(attr)) node.setAttribute(attr,String(Math.min(20,Math.max(1,parseInt(node.getAttribute(attr),10)||1))));
  if (node.hasAttribute('colwidth')) {
    const widths=node.getAttribute('colwidth').split(',').slice(0,20).map(n=>Math.min(1000,Math.max(60,parseInt(n,10)||60)));
    node.setAttribute('colwidth',widths.join(','));
  }
});
const clean = html => DOMPurify.sanitize(html, {
  ALLOWED_TAGS:allowedTags,
  ALLOWED_ATTR:['href','rel','style','colspan','rowspan','colwidth','data-type','data-checked','checked','type','disabled','start'],
  ALLOW_DATA_ATTR:false,
});
const escape = value => { const span=document.createElement('span'); span.textContent=value; return span.innerHTML; };
const plainHTML = value => value.split('\n').map(line=>`<p>${escape(line) || '<br>'}</p>`).join('');
const printCSS = 'body{max-width:850px;margin:36px auto;padding:0 20px;font:16px/1.6 Arial;color:#202b38}table{border-collapse:collapse;width:100%;table-layout:fixed}td,th{border:1px solid #aaa;padding:8px;overflow-wrap:anywhere}blockquote{border-left:3px solid #999;margin-left:0;padding-left:18px}pre{white-space:pre-wrap;background:#f1f3f5;padding:12px}h1,h2,h3{break-after:avoid}a{color:#2859ad}ul[data-type=taskList]{list-style:none;padding-left:0}li[data-type=taskItem]{display:flex;gap:8px}li[data-type=taskItem]>div{flex:1}li p{margin:0}';

function create(host, {onChange, onBeforeExport, getTitle, onMessage}) {
  host.innerHTML = `<div class="note-format-toolbar" role="toolbar" aria-label="Text formatting">
    <select data-format="block" aria-label="Paragraph style"><option value="paragraph">Normal text</option><option value="h1">Heading 1</option><option value="h2">Heading 2</option><option value="h3">Heading 3</option><option value="blockquote">Quote</option><option value="codeBlock">Code block</option></select>
    <button type="button" data-command="bold" aria-label="Bold (Ctrl+B)"><b>B</b></button><button type="button" data-command="italic" aria-label="Italic (Ctrl+I)"><i>I</i></button><button type="button" data-command="underline" aria-label="Underline (Ctrl+U)"><u>U</u></button><button type="button" data-command="strike" aria-label="Strikethrough"><s>S</s></button>
    <button type="button" data-command="highlight" aria-label="Highlight text">Highlight</button><button type="button" data-command="bulletList" aria-label="Bulleted list">Bullets</button><button type="button" data-command="orderedList" aria-label="Numbered list">1. List</button><button type="button" data-command="taskList" aria-label="Checklist in note">Checklist</button>
    <button type="button" data-command="undo" aria-label="Undo (Ctrl+Z)">Undo</button><button type="button" data-command="redo" aria-label="Redo (Ctrl+Shift+Z)">Redo</button>
    <details class="note-format-menu"><summary>More formatting</summary><div class="note-format-options">
    <label>Font<select data-format="font" aria-label="Font family"><option value="">Default</option>${fonts.map(f=>`<option>${f}</option>`).join('')}</select></label>
    <label>Size<select data-format="size" aria-label="Font size"><option value="">Default</option>${[12,14,16,18,20,24,28,32,36].map(n=>`<option value="${n}px">${n}</option>`).join('')}</select></label>
    <label>Text colour<select data-format="color" aria-label="Text colour"><option value="">Default</option><option value="#b91c1c">Red</option><option value="#166534">Green</option><option value="#1d4ed8">Blue</option><option value="#7e22ce">Purple</option><option value="#a16207">Gold</option></select></label>
    <label>Alignment<select data-format="align" aria-label="Text alignment"><option value="left">Left</option><option value="center">Centre</option><option value="right">Right</option><option value="justify">Justify</option></select></label>
    <button type="button" data-command="indent">Indent list</button><button type="button" data-command="outdent">Outdent list</button><button type="button" data-command="code">Inline code</button><button type="button" data-command="subscript">Subscript</button><button type="button" data-command="superscript">Superscript</button><button type="button" data-command="divider">Divider</button><button type="button" data-command="clear">Clear formatting</button></div></details>
    <details class="note-format-menu" data-menu="link"><summary>Link</summary><div class="note-format-options"><label>Link address<input data-link-url type="url" placeholder="https://example.com" aria-label="Link address"></label><button type="button" data-command="link">Apply link</button><button type="button" data-command="unlink">Remove link</button></div></details>
    <details class="note-format-menu" data-menu="table"><summary>Table</summary><div class="note-format-options"><label>Rows<input data-table-rows type="number" min="1" max="20" value="3"></label><label>Columns<input data-table-cols type="number" min="1" max="12" value="3"></label><button type="button" data-command="table">Insert table</button>
    <button type="button" data-table-command="addRowBefore">Row above</button><button type="button" data-table-command="addRowAfter">Row below</button><button type="button" data-table-command="addColumnBefore">Column before</button><button type="button" data-table-command="addColumnAfter">Column after</button><button type="button" data-table-command="deleteRow">Delete row</button><button type="button" data-table-command="deleteColumn">Delete column</button><button type="button" data-table-command="toggleHeaderRow">Toggle header</button><button type="button" data-table-command="mergeCells">Merge cells</button><button type="button" data-table-command="splitCell">Split cell</button><button type="button" data-table-command="deleteTable">Delete table</button></div></details>
    <details class="note-format-menu"><summary>Export</summary><div class="note-format-options"><button type="button" data-export="html">Download formatted note</button><button type="button" data-export="txt">Download text</button><button type="button" data-export="print">Print / PDF</button></div></details>
    </div><div class="note-document-host"></div><p class="note-word-count" aria-live="off"></p><p class="note-shortcuts">Ctrl/Cmd+B bold · I italic · U underline · Z undo. Select table cells to merge; Tab moves between cells.</p>`;
  const q = selector => host.querySelector(selector);
  let editor, loading = false, editable = true;
  const commands = {
    bold:'toggleBold', italic:'toggleItalic', underline:'toggleUnderline', strike:'toggleStrike', highlight:'toggleHighlight',
    bulletList:'toggleBulletList', orderedList:'toggleOrderedList', taskList:'toggleTaskList',
    code:'toggleCode', subscript:'toggleSubscript', superscript:'toggleSuperscript', undo:'undo', redo:'redo', divider:'setHorizontalRule',
  };
  function refresh() {
    if (!editor || editor.isDestroyed) return;
    for (const button of host.querySelectorAll('[data-command]')) {
      const command = button.dataset.command, mark = ['bold','italic','underline','strike','highlight','bulletList','orderedList','taskList','code','subscript','superscript'].includes(command);
      if (mark) button.setAttribute('aria-pressed',String(editor.isActive(command)));
      const method = commands[command];
      button.disabled = !editable || (method ? !editor.can()[method]() : false);
      if (['indent','outdent'].includes(command)) button.disabled = !editable || !editor.can()[command === 'indent' ? 'sinkListItem' : 'liftListItem'](editor.isActive('taskItem') ? 'taskItem' : 'listItem');
      if (command === 'unlink') button.disabled = !editable || !editor.isActive('link');
    }
    for (const button of host.querySelectorAll('[data-table-command]')) button.disabled = !editable || !editor.can()[button.dataset.tableCommand]();
    for (const input of host.querySelectorAll('select,input')) input.disabled = !editable;
    q('[data-format=block]').value = [1,2,3].find(n=>editor.isActive('heading',{level:n})) ? 'h'+editor.getAttributes('heading').level : editor.isActive('codeBlock') ? 'codeBlock' : editor.isActive('blockquote') ? 'blockquote' : 'paragraph';
    q('[data-format=align]').value = editor.getAttributes('paragraph').textAlign || editor.getAttributes('heading').textAlign || 'left';
    q('[data-format=font]').value = editor.getAttributes('textStyle').fontFamily || '';
    q('[data-format=size]').value = editor.getAttributes('textStyle').fontSize || '';
    q('[data-format=color]').value = editor.getAttributes('textStyle').color || '';
    const count = editor.storage.characterCount;
    q('.note-word-count').textContent = `${count.words()} words · ${count.characters().toLocaleString()} / 20,000 characters`;
  }
  editor = new Editor({
    element:q('.note-document-host'),
    extensions:[StarterKit.configure({heading:{levels:[1,2,3]}, link:{openOnClick:false,autolink:true,defaultProtocol:'https',isAllowedUri:url=>!!safeURL(url)}}),
      TableKit.configure({table:{resizable:true,cellMinWidth:60,renderWrapper:true}}), Highlight,
      TextAlign.configure({types:['heading','paragraph']}), TextStyle, Color, FontSize, FontFamily,
      TaskList, TaskItem.configure({nested:true}), CharacterCount.configure({limit:20000}), Subscript, Superscript],
    content:'<p></p>',
    editorProps:{attributes:{id:'ideaBody',role:'textbox','aria-label':'Note content','aria-multiline':'true','data-placeholder':'Write a note, or use the toolbar to add lists and tables...'},
      transformPastedHTML:clean,
      handleDOMEvents:{click:(_view,event)=>{ if(event.target.closest('a')) {event.preventDefault();return true;} return false; }}},
    onUpdate:()=>{ refresh(); if (!loading) onChange(); },
    onSelectionUpdate:refresh,
    onTransaction:()=>{ if (!loading) refresh(); },
  });
  host.querySelector('.note-format-toolbar').addEventListener('mousedown',event=>{if(event.target.closest('button'))event.preventDefault();});
  host.addEventListener('click', event => {
    const button = event.target.closest('button');
    if (!button || button.disabled) return;
    const command = button.dataset.command, table = button.dataset.tableCommand;
    if (button.dataset.export) { exportNote(button.dataset.export); return; }
    if (!editable) return;
    let chain = editor.chain().focus();
    if (commands[command]) chain[commands[command]]().run();
    else if (table) chain[table]().run();
    else if (command === 'clear') chain.unsetAllMarks().clearNodes().unsetTextAlign().run();
    else if (command === 'indent' || command === 'outdent') chain[command==='indent'?'sinkListItem':'liftListItem'](editor.isActive('taskItem')?'taskItem':'listItem').run();
    else if (command === 'table') {
      const rows=Number(q('[data-table-rows]').value),cols=Number(q('[data-table-cols]').value);
      if (!Number.isInteger(rows)||!Number.isInteger(cols)||rows<1||rows>20||cols<1||cols>12) { onMessage('Choose 1–20 rows and 1–12 columns.',true); return; }
      chain.insertTable({rows,cols,withHeaderRow:true}).run();
    } else if (command === 'link') {
      const href = safeURL(q('[data-link-url]').value.trim());
      if (!href) { onMessage('Use an https://, http:// or mailto: link.',true); return; }
      if (editor.state.selection.empty && !editor.isActive('link')) chain.insertContent({type:'text',text:href,marks:[{type:'link',attrs:{href}}]}).run();
      else chain.extendMarkRange('link').setLink({href}).run();
      q('[data-menu=link]').open = false;
    } else if (command === 'unlink') chain.extendMarkRange('link').unsetLink().run();
    refresh();
  });
  host.addEventListener('change',event=>{
    const format=event.target.dataset.format,value=event.target.value;
    if (!format || !editable) return;
    const chain=editor.chain().focus();
    if (format==='block') {
      if (/^h[123]$/.test(value)) chain.setHeading({level:Number(value[1])}).run();
      else if (value==='codeBlock') chain.toggleCodeBlock().run();
      else if (value==='blockquote') chain.toggleBlockquote().run();
      else chain.clearNodes().setParagraph().run();
    } else if (format==='align') chain.setTextAlign(value).run();
    else if (format==='size') value ? chain.setFontSize(value).run() : chain.unsetFontSize().run();
    else if (format==='font') value ? chain.setFontFamily(value).run() : chain.unsetFontFamily().run();
    else if (format==='color') value ? chain.setColor(value).run() : chain.unsetColor().run();
    refresh();
  });
  q('[data-menu=link]').addEventListener('toggle',()=>{if(q('[data-menu=link]').open)q('[data-link-url]').value=editor.getAttributes('link').href || '';});
  function exportNote(format) {
    if (!onBeforeExport()) return;
    const title=getTitle() || 'Untitled note',html=`<!doctype html><html lang="en"><head><meta charset="utf-8"><title>${escape(title)}</title><style>${printCSS}</style></head><body><h1>${escape(title)}</h1>${clean(editor.getHTML())}</body></html>`;
    if (format==='print') {
      const frame=document.createElement('iframe'); frame.className='note-print-frame'; frame.title='Print current note';
      frame.onload=()=>{frame.contentWindow.focus();frame.contentWindow.print();};
      frame.srcdoc=html;document.body.append(frame);setTimeout(()=>frame.remove(),60000);return;
    }
    const blob=new Blob([format==='txt'?title+'\n\n'+editor.getText():html],{type:format==='txt'?'text/plain;charset=utf-8':'text/html;charset=utf-8'}),url=URL.createObjectURL(blob),link=document.createElement('a');
    link.href=url;link.download=(title.replace(/[^a-z0-9 _-]/gi,'').trim().slice(0,80)||'note')+'.'+format;link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
  }
  refresh();
  return {
    setContent(note) {
      loading=true;
      // Recreate state so undo never reaches another note's content.
      const content=note.bodyHtml === undefined ? plainHTML(note.body || '') : clean(note.bodyHtml);
      editor.commands.setContent(content,{emitUpdate:false});
      const state=editor.state;
      editor.view.updateState(state.constructor.create({schema:state.schema,doc:state.doc,plugins:state.plugins}));
      loading=false;refresh();
    },
    getContent:()=>({body:editor.getText({blockSeparator:'\n'}),bodyHtml:clean(editor.getHTML())}),
    hasContent:()=>!editor.isEmpty,
    setEditable(value) { if (value!==editable) {editable=value;editor.setEditable(value,false);} refresh(); },
    destroy:()=>{if(!editor.isDestroyed)editor.destroy();},
  };
}
window.QuantNoteEditor={create};
