// Drawings use bar timestamps and prices, so edits survive zooming and reloads.
const drawingDefaults = {color:'#f5c451', width:2, dash:'solid', transparency:0, fillColor:'#f5c451', fillTransparency:88};
let drawingStyle = {...drawingDefaults}, selectedDrawing = null, drawingUndo = [], drawingRedo = [], drawingGesture = null, styleBaseline = null;
try { drawingStyle = {...drawingDefaults, ...JSON.parse(localStorage.getItem('atlas.drawingStyle') || '{}')}; } catch {}
const drawingNames = {trend:'Trend line', ray:'Ray', horizontal:'Horizontal line', vertical:'Vertical line', rectangle:'Rectangle', fib:'Fibonacci', measure:'Measurement'};
const drawingEditor = document.createElement('div');
drawingEditor.className = 'drawing-editor';
drawingEditor.innerHTML = `
  <div class="drawing-style-row" role="group" aria-label="Drawing appearance">
    <strong id="drawingSelection">New drawing</strong>
    <label>Line <input id="drawingColor" type="color" aria-label="Drawing line color"></label>
    <label>Width <select id="drawingWidth" aria-label="Drawing line width"><option value="1">1 px</option><option value="2">2 px</option><option value="3">3 px</option><option value="4">4 px</option><option value="6">6 px</option></select></label>
    <label>Style <select id="drawingDash" aria-label="Drawing line style"><option value="solid">Solid</option><option value="dashed">Dashed</option><option value="dotted">Dotted</option></select></label>
    <label>Line transparency <input id="drawingTransparency" aria-label="Drawing line transparency" type="range" min="0" max="100"><output id="drawingTransparencyValue"></output></label>
    <label>Fill <input id="drawingFillColor" type="color" aria-label="Drawing fill color"></label>
    <label>Fill transparency <input id="drawingFillTransparency" aria-label="Drawing fill transparency" type="range" min="0" max="100"><output id="drawingFillTransparencyValue"></output></label>
  </div>
  <div class="drawing-action-row">
    <span class="drawing-help">Select a shape to move it. Drag its handles to resize.</span>
    <label id="drawingPriceALabel" hidden>Price A <input id="drawingPriceA" type="number" step="any" aria-label="Drawing price A"></label>
    <label id="drawingPriceBLabel" hidden>Price B <input id="drawingPriceB" type="number" step="any" aria-label="Drawing price B"></label>
    <button id="lockDrawing" disabled>Lock</button><button id="hideDrawing" disabled>Hide</button><button id="duplicateDrawing" disabled>Duplicate</button><button id="deleteDrawing" disabled>Delete</button>
    <details id="drawingObjects"><summary>Objects <span id="drawingObjectCount">0</span></summary><div id="drawingObjectList" aria-label="Chart drawings"></div></details>
  </div>`;
document.querySelector('.range-toolbar').before(drawingEditor);
$('chart').tabIndex = 0;

function drawingKey() { return selected ? `atlas.drawings.${selected.symbol}.${interval}` : ''; }
function drawingSnapshot() { return JSON.stringify(drawings); }
function selectedShape() { return drawings.find(d => d.id === selectedDrawing); }
function normalizeDrawing(d) {
  if (!d || !drawingNames[d.type]) return null;
  if (d.type === 'horizontal' ? !Number.isFinite(d.price) : d.type === 'vertical' ? !Number.isFinite(Date.parse(d.date)) : ![d.a,d.b].every(a => a && Number.isFinite(a.price) && Number.isFinite(Date.parse(a.date)))) return null;
  return {...d, id:d.id || crypto.randomUUID(), style:{...drawingDefaults,...d.style}};
}
function loadDrawings() {
  try { const stored=JSON.parse(localStorage.getItem(drawingKey()) || '[]'); drawings=Array.isArray(stored)?stored.map(normalizeDrawing).filter(Boolean):[]; } catch { drawings=[]; }
  selectedDrawing=null; pending=null; drawingGesture=null; drawingUndo=[]; drawingRedo=[]; styleBaseline=null; drag=null;
  syncDrawingEditor();
}
function saveDrawings() {
  if (drawingKey()) try { localStorage.setItem(drawingKey(),drawingSnapshot()); } catch { showError('Browser storage is full. Drawing changes could not be saved.'); }
}
function rememberDrawing(before) {
  if(before === drawingSnapshot()) return;
  drawingUndo.push(before); if(drawingUndo.length>80)drawingUndo.shift(); drawingRedo=[];
}
function commitDrawing(before) { rememberDrawing(before); saveDrawings(); syncDrawingEditor(); draw(); }
function finishStyleEdit() { if(styleBaseline!==null){rememberDrawing(styleBaseline);styleBaseline=null;syncDrawingEditor();} }
function restoreDrawing(direction) {
  finishStyleEdit(); cancelDrawingGesture();
  const from=direction==='undo'?drawingUndo:drawingRedo,to=direction==='undo'?drawingRedo:drawingUndo;
  if(!from.length)return;
  to.push(drawingSnapshot()); drawings=JSON.parse(from.pop()); if(!selectedShape())selectedDrawing=null;
  pending=null; saveDrawings(); syncDrawingEditor(); draw();
}
function syncDrawingEditor() {
  const d=selectedShape(), style=d?.style || drawingStyle, locked=!!d?.locked;
  $('drawingSelection').textContent=d?`${drawingNames[d.type]}${locked?' · Locked':''}`:'New drawing';
  const fields={drawingColor:'color',drawingWidth:'width',drawingDash:'dash',drawingTransparency:'transparency',drawingFillColor:'fillColor',drawingFillTransparency:'fillTransparency'};
  for(const [id,key] of Object.entries(fields)){ $(id).value=style[key]; $(id).disabled=locked; }
  $('drawingTransparencyValue').textContent=`${style.transparency}%`;
  $('drawingFillTransparencyValue').textContent=`${style.fillTransparency}%`;
  for(const id of ['lockDrawing','hideDrawing','duplicateDrawing','deleteDrawing'])$(id).disabled=!d;
  $('lockDrawing').textContent=locked?'Unlock':'Lock';$('lockDrawing').setAttribute('aria-pressed',locked);
  $('hideDrawing').textContent=d?.hidden?'Show':'Hide';$('hideDrawing').setAttribute('aria-pressed',!!d?.hidden);
  $('drawingPriceALabel').hidden=!d||d.type==='vertical';$('drawingPriceBLabel').hidden=!d?.b;
  $('drawingPriceA').value=d?(d.price??d.a?.price??''):'';$('drawingPriceB').value=d?.b?.price??'';
  $('drawingPriceA').disabled=$('drawingPriceB').disabled=locked;
  $('undoDrawing').disabled=!drawingUndo.length;$('redoDrawing').disabled=!drawingRedo.length;$('clearDrawings').disabled=!drawings.length;
  $('drawingObjectCount').textContent=drawings.length;
  $('drawingObjectList').innerHTML=drawings.map((shape,i)=>`<button data-select-drawing="${esc(shape.id)}" class="${shape.id===selectedDrawing?'active':''}"><i style="background:${/^#[0-9a-f]{6}$/i.test(shape.style.color)?shape.style.color:drawingDefaults.color}"></i>${i+1}. ${drawingNames[shape.type]}${shape.locked?' · Locked':''}${shape.hidden?' · Hidden':''}</button>`).join('')||'<span>No drawings yet.</span>';
  updateDrawingHint();
}
const styleFields={drawingColor:'color',drawingWidth:'width',drawingDash:'dash',drawingTransparency:'transparency',drawingFillColor:'fillColor',drawingFillTransparency:'fillTransparency'};
for(const [id,key] of Object.entries(styleFields)) {
  $(id).addEventListener('input',()=>{
    const d=selectedShape();if(d?.locked)return;
    if(d && styleBaseline===null)styleBaseline=drawingSnapshot();
    const value=['width','transparency','fillTransparency'].includes(key)?Number($(id).value):$(id).value;
    drawingStyle[key]=value;if(d)d.style[key]=value;
    try{localStorage.setItem('atlas.drawingStyle',JSON.stringify(drawingStyle));}catch{}
    if(d)saveDrawings();
    $('drawingTransparencyValue').textContent=`${(d?.style||drawingStyle).transparency}%`;
    $('drawingFillTransparencyValue').textContent=`${(d?.style||drawingStyle).fillTransparency}%`;draw();
  });
  $(id).addEventListener('change',finishStyleEdit);
}
for(const [id,key] of [['drawingPriceA','a'],['drawingPriceB','b']])$(id).onchange=()=>{
  const d=selectedShape(),value=Number($(id).value);if(!d||d.locked||!$(id).value||!Number.isFinite(value))return syncDrawingEditor();
  const before=drawingSnapshot();if(d.type==='horizontal')d.price=value;else if(d[key])d[key].price=value;commitDrawing(before);
};
function deleteSelectedDrawing(){finishStyleEdit();if(!selectedShape())return;const before=drawingSnapshot();drawings=drawings.filter(d=>d.id!==selectedDrawing);selectedDrawing=null;commitDrawing(before);}
$('deleteDrawing').onclick=deleteSelectedDrawing;
$('lockDrawing').onclick=()=>{finishStyleEdit();const d=selectedShape();if(!d)return;const before=drawingSnapshot();d.locked=!d.locked;commitDrawing(before);};
$('hideDrawing').onclick=()=>{finishStyleEdit();const d=selectedShape();if(!d)return;const before=drawingSnapshot();d.hidden=!d.hidden;commitDrawing(before);};
$('duplicateDrawing').onclick=()=>{finishStyleEdit();const d=selectedShape();if(!d)return;const before=drawingSnapshot(),copy=structuredClone(d);copy.id=crypto.randomUUID();copy.locked=false;copy.hidden=false;const offset=geometry().span*.025;if(copy.a){copy.a.price+=offset;copy.b.price+=offset;}else if(copy.type==='horizontal')copy.price+=offset;else{const i=barIndex(copy.date);copy.date=rows[Math.min(rows.length-1,Math.round(i)+2)]?.date||copy.date;}drawings.push(copy);selectedDrawing=copy.id;commitDrawing(before);};
$('drawingObjectList').onclick=e=>{const button=e.target.closest('[data-select-drawing]');if(button){finishStyleEdit();selectedDrawing=button.dataset.selectDrawing;tool='cursor';pending=null;syncToolButtons();syncDrawingEditor();draw();}};
$('undoDrawing').onclick=()=>restoreDrawing('undo');$('redoDrawing').onclick=()=>restoreDrawing('redo');
$('clearDrawings').onclick=()=>{finishStyleEdit();const before=drawingSnapshot();drawings=[];pending=null;selectedDrawing=null;commitDrawing(before);};

function barIndex(date) {
  if(!rows.length)return 0;
  let low=0,high=rows.length-1;const target=Date.parse(date);
  while(low<=high){const middle=(low+high)>>1,value=Date.parse(rows[middle].date);if(value===target)return middle;if(value<target)low=middle+1;else high=middle-1;}
  if(low>0&&low<rows.length){const a=Date.parse(rows[low-1].date),b=Date.parse(rows[low].date);return low-1+(target-a)/(b-a);}
  const edge=low===0?0:rows.length-1,other=low===0?Math.min(1,rows.length-1):Math.max(0,rows.length-2);
  const step=Math.abs(Date.parse(rows[edge].date)-Date.parse(rows[other].date))||(interval==='1d'?86400000:60000);
  return edge+(target-Date.parse(rows[edge].date))/step;
}
function anchorPoint(g,a) {const i=barIndex(a.date)-viewStart;return{x:g.x(i),y:g.y(a.price),i};}
function drawingHandles(g,d) {
  if(d.type==='horizontal')return [{x:g.left+g.pw/2,y:g.y(d.price),key:'price'}];
  if(d.type==='vertical')return [{x:anchorPoint(g,{date:d.date,price:0}).x,y:g.top+g.ph/2,key:'date'}];
  const a=anchorPoint(g,d.a),b=anchorPoint(g,d.b),points=[{...a,key:'a'},{...b,key:'b'}];
  if(d.type==='rectangle'||d.type==='measure')points.push({x:a.x,y:b.y,key:'ab'},{x:b.x,y:a.y,key:'ba'});
  return points;
}
function drawingRayEnd(g,a,b) {
  const dx=b.x-a.x,dy=b.y-a.y;if(!dx&&!dy)return b;
  const steps=[];if(dx>0)steps.push((g.left+g.pw-a.x)/dx);if(dx<0)steps.push((g.left-a.x)/dx);if(dy>0)steps.push((g.top+g.ph-a.y)/dy);if(dy<0)steps.push((g.top-a.y)/dy);
  const t=Math.max(1,Math.min(...steps.filter(n=>n>=0)));return{x:a.x+dx*t,y:a.y+dy*t};
}
function strokeDistance(p,a,b) {const dx=b.x-a.x,dy=b.y-a.y,t=Math.max(0,Math.min(1,((p.x-a.x)*dx+(p.y-a.y)*dy)/(dx*dx+dy*dy||1)));return Math.hypot(p.x-a.x-t*dx,p.y-a.y-t*dy);}
function hitDrawing(g,px,py) {
  const p={x:px,y:py},ordered=[...drawings].reverse(),current=selectedShape();
  if(current&&!current.hidden&&!current.locked)for(const handle of drawingHandles(g,current))if(Math.hypot(px-handle.x,py-handle.y)<10)return{drawing:current,handle:handle.key};
  for(const d of ordered){if(d.hidden)continue;let hit=false;
    if(d.type==='horizontal')hit=Math.abs(py-g.y(d.price))<7;
    else if(d.type==='vertical')hit=Math.abs(px-anchorPoint(g,{date:d.date,price:0}).x)<7;
    else{const a=anchorPoint(g,d.a),b=anchorPoint(g,d.b);
      if(d.type==='rectangle'||d.type==='measure')hit=px>=Math.min(a.x,b.x)-6&&px<=Math.max(a.x,b.x)+6&&py>=Math.min(a.y,b.y)-6&&py<=Math.max(a.y,b.y)+6;
      else if(d.type==='fib')hit=[0,.236,.382,.5,.618,.786,1].some(level=>strokeDistance(p,{x:a.x,y:g.y(d.a.price+(d.b.price-d.a.price)*level)},{x:b.x,y:g.y(d.a.price+(d.b.price-d.a.price)*level)})<7);
      else hit=strokeDistance(p,a,d.type==='ray'?drawingRayEnd(g,a,b):b)<7;
    }
    if(hit)return{drawing:d,handle:null};
  }return null;
}
function drawShape(c,g,d,preview=false) {
  if(d.hidden)return;
  const s={...drawingDefaults,...(d.style||drawingStyle)},alpha=1-s.transparency/100;
  c.save();c.beginPath();c.rect(g.left,g.top,g.pw,g.ph);c.clip();
  c.strokeStyle=s.color;c.lineWidth=Number(s.width);c.globalAlpha=alpha*(preview?.65:1);
  c.setLineDash(preview?[5,4]:s.dash==='dashed'?[8,5]:s.dash==='dotted'?[2,4]:[]);
  if(d.type==='horizontal'){const y=g.y(d.price);c.beginPath();c.moveTo(g.left,y);c.lineTo(g.left+g.pw,y);c.stroke();c.fillStyle=s.color;c.fillText(fmt(d.price),g.left+8,y-6);}
  else if(d.type==='vertical'){const x=anchorPoint(g,{date:d.date,price:g.min}).x;c.beginPath();c.moveTo(x,g.top);c.lineTo(x,g.top+g.ph);c.stroke();c.fillStyle=s.color;c.fillText(labelTime(d.date),x+6,g.top+12);}
  else {const a=anchorPoint(g,d.a),b=anchorPoint(g,d.b);
    if(d.type==='rectangle'||d.type==='measure'){
      const x=Math.min(a.x,b.x),y=Math.min(a.y,b.y),w=Math.abs(b.x-a.x),h=Math.abs(b.y-a.y);
      c.globalAlpha=(1-s.fillTransparency/100)*(preview?.65:1);c.fillStyle=s.fillColor;c.fillRect(x,y,w,h);c.globalAlpha=alpha;c.strokeRect(x,y,w,h);
      if(d.type==='measure'){c.fillStyle=s.color;const change=d.a.price?(d.b.price/d.a.price-1)*100:0,bars=Math.round(Math.abs(barIndex(d.b.date)-barIndex(d.a.date)))+1;c.fillText(`${pct(change)} · ${bars} bars`,x+5,Math.max(g.top+12,y-5));}
    }else if(d.type==='fib'){
      for(const level of [0,.236,.382,.5,.618,.786,1]){const price=d.a.price+(d.b.price-d.a.price)*level,y=g.y(price);c.beginPath();c.moveTo(a.x,y);c.lineTo(b.x,y);c.stroke();c.fillStyle=s.color;c.fillText(`${level.toFixed(3)}  ${fmt(price)}`,Math.min(a.x,b.x)+5,y-4);}
    }else{const end=d.type==='ray'?drawingRayEnd(g,a,b):b;c.beginPath();c.moveTo(a.x,a.y);c.lineTo(end.x,end.y);c.stroke();}
  }
  if(d.id===selectedDrawing||preview){c.globalAlpha=1;c.setLineDash([]);c.strokeStyle=d.locked?'#8b98aa':'#eaf2ff';c.fillStyle=s.color;c.lineWidth=2;for(const p of drawingHandles(g,d)){c.beginPath();c.arc(p.x,p.y,5,0,Math.PI*2);c.fill();c.stroke();}}
  c.restore();
}
function drawAnnotations(c,g) {
  for(const d of drawings)drawShape(c,g,d);
  if(pending&&hoverAnchor)drawShape(c,g,{type:tool,a:pending,b:hoverAnchor,style:drawingStyle},true);
  if(pending){const p=anchorPoint(g,pending);c.fillStyle=drawingStyle.color;c.beginPath();c.arc(p.x,p.y,5,0,Math.PI*2);c.fill();}
}
function pointer(event,snap=tool!=='cursor',g=geometry()) {
  const px=event.clientX-g.box.left,py=event.clientY-g.box.top,i=Math.max(0,Math.min(g.data.length-1,Math.floor((px-g.left)/g.pw*g.data.length))),bar=g.data[i];
  let price=g.price(Math.max(g.top,Math.min(g.top+g.ph,py))),label=null;
  if(magnet&&snap&&!event.shiftKey&&bar){const candidates=[['O',bar.open],['H',bar.high],['L',bar.low],['C',bar.close]];[label,price]=candidates.reduce((best,item)=>Math.abs(g.y(item[1])-py)<Math.abs(g.y(best[1])-py)?item:best);}
  return{g,px,py,i,inside:px>=g.left&&px<=g.left+g.pw&&py>=g.top&&py<=g.top+g.ph,anchor:{date:bar?.date,price,snap:label}};
}
function updateDrawingHint() {
  let text=tool==='cursor'?(selectedShape()?'Drag shape to move · Drag handles to resize':'Select drawings · Drag empty chart to pan'):`${drawingNames[tool]} · ${pending?'click second point':'click to place'}`;
  if(tool!=='cursor'&&magnet)text+=' · Shift = free placement';$('drawingHint').textContent=text;$('drawingHint').title=text;
}
function syncToolButtons() {for(const b of $('drawingTools').querySelectorAll('[data-tool]'))b.classList.toggle('active',b.dataset.tool===tool);$('chart').classList.toggle('crosshair',tool!=='cursor');updateDrawingHint();}
function setTool(value) {finishStyleEdit();cancelDrawingGesture();tool=value;pending=null;if(value!=='cursor')selectedDrawing=null;syncToolButtons();syncDrawingEditor();draw();}
function cancelDrawingGesture() {if(drawingGesture){drawings=JSON.parse(drawingGesture.before);drawingGesture=null;saveDrawings();}drag=null;}
function finishDrawingGesture() {if(drawingGesture){const before=drawingGesture.before;drawingGesture=null;commitDrawing(before);}drag=null;}
$('drawingTools').onclick=e=>{const button=e.target.closest('[data-tool]');if(button)setTool(button.dataset.tool);};
$('magnet').onclick=()=>{magnet=!magnet;$('magnet').classList.toggle('active',magnet);$('magnet').setAttribute('aria-pressed',magnet);updateDrawingHint();draw();};
$('chart').addEventListener('pointerdown',e=>{
  if(!rows.length||e.button!==0)return;const p=pointer(e);if(!p.inside)return;e.preventDefault();$('chart').focus({preventScroll:true});finishStyleEdit();$('chart').setPointerCapture(e.pointerId);
  if(tool==='cursor'){
    const hit=hitDrawing(p.g,p.px,p.py);selectedDrawing=hit?.drawing.id||null;
    if(hit&&!hit.drawing.locked)drawingGesture={before:drawingSnapshot(),original:structuredClone(hit.drawing),handle:hit.handle,x:e.clientX,y:e.clientY,g:p.g};
    else if(!hit)drag={x:e.clientX,start:viewStart,moved:false};
    syncDrawingEditor();draw();return;
  }
  const before=drawingSnapshot();let shape;
  if(tool==='horizontal')shape={type:tool,price:p.anchor.price};
  else if(tool==='vertical')shape={type:tool,date:p.anchor.date};
  else if(!pending){pending=p.anchor;hoverAnchor=p.anchor;updateDrawingHint();draw();return;}
  else shape={type:tool,a:pending,b:p.anchor};
  shape=normalizeDrawing({...shape,style:{...drawingStyle}});drawings.push(shape);selectedDrawing=shape.id;pending=null;tool='cursor';syncToolButtons();commitDrawing(before);
});
window.addEventListener('pointermove',e=>{
  if(drawingGesture){
    const edit=drawingGesture,d=selectedShape();if(!d)return;const p=pointer(e,!!edit.handle,edit.g),original=edit.original;
    if(edit.handle){
      if(edit.handle==='price')d.price=p.anchor.price;
      else if(edit.handle==='date')d.date=p.anchor.date;
      else if(edit.handle==='a'||edit.handle==='b')d[edit.handle]={...p.anchor};
      else if(edit.handle==='ab'){d.a.date=p.anchor.date;d.b.price=p.anchor.price;}
      else {d.b.date=p.anchor.date;d.a.price=p.anchor.price;}
    }else{
      let delta=Math.round((e.clientX-edit.x)/edit.g.pw*edit.g.data.length);const movedPrice=price=>edit.g.price(edit.g.y(price)+(e.clientY-edit.y));
      const anchors=original.a?[original.a,original.b]:original.type==='vertical'?[{date:original.date}]:[];
      const indexes=anchors.map(a=>Math.round(barIndex(a.date)));
      const outside=indexes.some(i=>i<0||i>=rows.length);
      if(outside)delta=0;
      else if(indexes.length)delta=Math.max(-Math.min(...indexes),Math.min(rows.length-1-Math.max(...indexes),delta));
      if(original.a){for(const key of ['a','b'])d[key]={...original[key],date:delta===0?original[key].date:rows[Math.round(barIndex(original[key].date))+delta].date,price:movedPrice(original[key].price)};}
      else if(original.type==='horizontal')d.price=movedPrice(original.price);
      else d.date=delta===0?original.date:rows[indexes[0]+delta].date;
    }draw();return;
  }
  if(drag){const g=geometry(),shift=Math.round((drag.x-e.clientX)/g.pw*viewCount);drag.moved=drag.moved||Math.abs(e.clientX-drag.x)>3;viewStart=Math.max(0,Math.min(rows.length-viewCount,drag.start+shift));draw();return;}
  if(!rows.length)return;const p=pointer(e);if(!p.inside)return;
  hover=p.i;hoverAnchor=p.anchor;hoverY=tool!=='cursor'&&magnet&&!e.shiftKey?p.g.y(p.anchor.price):p.py;
  const hit=tool==='cursor'?hitDrawing(p.g,p.px,p.py):null;$('chart').style.cursor=tool!=='cursor'?'crosshair':hit?(hit.drawing.locked?'default':hit.handle?'crosshair':'move'):'grab';
  const r=p.g.data[hover];if(r)$('ohlc').textContent=`O ${fmt(r.open)}  H ${fmt(r.high)}  L ${fmt(r.low)}  C ${fmt(r.close)}  Vol ${fmt(r.volume,0)}`;draw();
});
window.addEventListener('pointerup',finishDrawingGesture);
$('chart').addEventListener('pointercancel',()=>{cancelDrawingGesture();syncDrawingEditor();draw();});
$('chart').addEventListener('lostpointercapture',finishDrawingGesture);
window.addEventListener('blur',finishDrawingGesture);
$('chart').addEventListener('pointerleave',()=>{if(!drag&&!drawingGesture){hover=-1;hoverY=null;hoverAnchor=null;draw();}});
$('chart').addEventListener('wheel',e=>{
  if(!rows.length)return;e.preventDefault();if(drawingGesture||drag)return;
  const old=viewCount;viewCount=Math.max(Math.min(20,rows.length),Math.min(rows.length,Math.round(old*(e.deltaY>0?1.2:.8))));
  const ratio=Math.max(0,Math.min(1,(e.offsetX-10)/Math.max(1,e.currentTarget.clientWidth-82))),focus=viewStart+ratio*old;
  viewStart=Math.max(0,Math.min(rows.length-viewCount,Math.round(focus-ratio*viewCount)));draw();
},{passive:false});
new ResizeObserver(()=>draw()).observe($('chart'));
document.addEventListener('keydown',e=>{
  const editing=['INPUT','SELECT','TEXTAREA'].includes(document.activeElement.tagName)||document.activeElement.isContentEditable,dialog=document.querySelector('dialog[open]');
  if(editing||dialog)return;
  if(e.key==='Escape'){cancelDrawingGesture();pending=null;selectedDrawing=null;setTool('cursor');}
  else if((e.ctrlKey||e.metaKey)&&e.key.toLowerCase()==='z'){e.preventDefault();restoreDrawing(e.shiftKey?'redo':'undo');}
  else if((e.ctrlKey||e.metaKey)&&e.key.toLowerCase()==='y'){e.preventDefault();restoreDrawing('redo');}
  else if((e.key==='Delete'||e.key==='Backspace')&&selectedShape()){e.preventDefault();deleteSelectedDrawing();}
  else if(e.key==='/'){e.preventDefault();$('search').focus();}
});
syncDrawingEditor();
