import * as THREE from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
import {GLTFLoader} from 'three/addons/loaders/GLTFLoader.js';
import {cleanThicknesses,setThickness} from './preserve-settings.mjs';

const $ = id => document.getElementById(id);
const reviewJob = new URLSearchParams(location.search).get('review');
const DESIGN_VERSION = 12;
const settings = ['surfaceMode','surfacePattern','surfaceThickness','surfaceCell','surfaceWidth','poreRounding','junctionBlend','edgeWidth','surfaceContinuity','seamStyle','latticeType','cell','wall','axis','layer','scope','unselected','depth','shell','pitch','smooth','memoryMode'];
const state = {token:'', model:null, bodies:[], body:0, role:'design', design:new Set(), preserve:new Set(), busy:false, job:null, result:null, dirty:true, view:'source', source:null, resultGroup:null, bounds:null, centre:new THREE.Vector3(), radius:30, restoring:false};
state.faceShells={};
const descriptions = {
 kelvin:['空间连通的截角八面体杆网。具有六边形与四边形孔面，不是六棱柱挤出。','单元大小为三轴方向的立方重复周期。'],
 hex3d:['逐层六边形杆框 + 顶点竖向连杆，形成开放的六棱柱空间蜂窝。','单元大小为六边形中心距（对边距）；层高独立设置。'],
 hex:['六边形蜂窝壁沿所选轴向贯通。其厚度为壁板厚度，不是圆杆直径。','单元大小为六边形中心距（对边距）。'],
 hexcell:['六边形棱柱壁 + 水平隔板。腔体封闭，需评估排粉和排液。','单元大小为六边形中心距（对边距）。'],
 bcc:['体心至八角点的斜向圆杆，在三维空间重复连接。','单元大小为立方晶胞边长。'],
 cubic:['沿立方体十二条边布置的三轴圆杆结构。','单元大小为立方晶胞边长。'],
 grid:['三组互相垂直的连续壁板，形成封闭立方腔。','单元大小为立方晶胞边长。'],
 gyroid:['连续弯曲的 Gyroid 薄壁曲面，非六边形杆网。','单元大小为三轴重复周期；壁厚为近似值。'],
 diamond:['Diamond 双连续曲面形成的薄壁结构。','单元大小为三轴重复周期；壁厚为近似值。'],
 schwarz:['Schwarz-P 曲面，三轴方向存在贯通孔道。','单元大小为三轴重复周期；壁厚为近似值。'],
 iwp:['I-WP 连续薄壁曲面，局部壁厚会受曲率影响。','单元大小为三轴重复周期；壁厚为近似值。']
};
const struts = new Set(['kelvin','hex3d','bcc','cubic']);
const polygons = {
 kelvin:'<path d="M25 8 47 8 61 23 61 45 46 60 24 60 9 45 9 23Z M25 8 31 23 24 38 9 45 M47 8 43 23 51 38 61 45 M24 60 24 38 31 23 43 23 51 38 46 60 M24 38 38 45 51 38 M38 45 38 60 M9 23 31 23 M43 23 61 23"/>',
 hex3d:'<path d="M18 10 38 10 48 27 38 44 18 44 8 27Z M32 24 52 24 62 41 52 58 32 58 22 41Z M18 10 32 24 M38 10 52 24 M48 27 62 41 M38 44 52 58 M18 44 32 58 M8 27 22 41"/>',
 bcc:'<path d="M12 20 43 9 67 25 35 37Z M12 20v30l23 15 32-12V25 M35 37v28 M12 20l27 19 4-30 M67 25 39 39 28 53 M12 50l27-11 28 14 M35 65l4-26-4-2"/>',
 hex:'<path d="M17 8 34 8 43 23 34 38 17 38 8 23Z M43 23 60 23 69 38 60 53 43 53 34 38 M17 38 8 53 17 68 34 68 43 53 M43 23 60 23 69 8"/>',
 cubic:'<path d="M12 20 43 9 67 25 35 37Z M12 20v30l23 15 32-12V25 M35 37v28 M27 15v30l24 15 M55 17 24 30v28 M12 35l23 16 32-12"/>'
};
let renderer, scene, camera, controls, grid, sourceGroup, resultRoot, clipPlane;
try {
 renderer = new THREE.WebGLRenderer({antialias:true,alpha:true});
 renderer.setPixelRatio(Math.min(devicePixelRatio,2)); renderer.localClippingEnabled = true;
 renderer.setClearColor(0xf2f5f4,1); renderer.outputColorSpace = THREE.SRGBColorSpace;
 renderer.toneMapping = THREE.ACESFilmicToneMapping; renderer.toneMappingExposure = 1.35;
 $('viewport').append(renderer.domElement);
 scene = new THREE.Scene();
 camera = new THREE.PerspectiveCamera(34,1,.01,10000); camera.up.set(0,0,1);
 controls = new OrbitControls(camera,renderer.domElement); controls.enableDamping = true; controls.dampingFactor=.09;
 scene.add(new THREE.HemisphereLight(0xffffff,0x738780,2));
 const light = new THREE.DirectionalLight(0xffffff,3); light.position.set(-80,-120,150); scene.add(light);
 const fill = new THREE.DirectionalLight(0xd1e1dc,1.8); fill.position.set(70,30,60); scene.add(fill);
 sourceGroup = new THREE.Group(); resultRoot = new THREE.Group(); scene.add(sourceGroup,resultRoot);
 clipPlane = new THREE.Plane(new THREE.Vector3(0,0,-1),0);
 const resize = () => {const rect=$('viewport').getBoundingClientRect(); renderer.setSize(rect.width,rect.height); camera.aspect=rect.width/rect.height;camera.updateProjectionMatrix();};
 new ResizeObserver(resize).observe($('viewport')); resize();
 renderer.setAnimationLoop(()=>{controls.update();renderer.render(scene,camera);});
 $('viewEmpty').hidden=true;
} catch(err) {
 $('viewEmpty').textContent='三维显示不可用：请启用浏览器 WebGL / 硬件加速。';
 throw err;
}

async function api(path,data,raw=false) {
 const opts=data===undefined?{}:{method:'POST',headers:{'X-Lattice-Token':state.token,'Content-Type':raw?'application/octet-stream':'application/json'},body:raw?data:JSON.stringify(data)};
 const r=await fetch(path,opts); if(!r.ok) throw new Error(`请求失败 (${r.status})`);
 const value=await r.json(); if(value.error) throw new Error(value.error); return value;
}
function message(text,error=false){$('messages').textContent=text;$('messages').style.color=error?'#b46842':'';}
function remember(){
 if(!state.model||state.restoring||reviewJob)return;
 try{localStorage.setItem('latticeStudio.session',JSON.stringify({version:DESIGN_VERSION,model:state.model,body:state.body,filename:$('filename').textContent,settings:Object.fromEntries(settings.map(id=>[id,$(id).value])),design:[...state.design],preserve:[...state.preserve],faceShells:cleanThicknesses(state.faceShells,state.preserve),result:state.result?.id,dirty:state.dirty}));}catch{}
}
function config(quality='preview') {
 return {surface_mode:$('surfaceMode').value,surface_pattern:$('surfacePattern').value,surface_thickness:Number($('surfaceThickness').value),surface_cell:Number($('surfaceCell').value),surface_width:Number($('surfaceWidth').value),pore_rounding:Number($('poreRounding').value),junction_blend:Number($('junctionBlend').value),edge_width:Number($('edgeWidth').value),surface_continuity:$('surfaceContinuity').value,seam_style:$('seamStyle').value,lattice:$('latticeType').value,cell:Number($('cell').value),wall:Number($('wall').value),axis:$('axis').value,layer:Number($('layer').value),scope:$('scope').value,unselected:$('unselected').value,design_version:DESIGN_VERSION,depth:Number($('depth').value),shell:Number($('shell').value),preserve_thicknesses:cleanThicknesses(state.faceShells,state.preserve),pitch:Number($('pitch').value),smooth:Number($('smooth').value),memory_mode:$('memoryMode').value,design_faces:[...state.design],preserve_faces:[...state.preserve],quality};
}
function request(quality){return {model:state.model,body:state.body,config:config(quality)};}
function busy(value){
 state.busy=value;
 for(const id of [...settings,'openImport','clearFaces','roleDesign','rolePreserve','bodySelect']) $(id).disabled=value || (id==='bodySelect'&&!state.model);
 $('preview').disabled=value||!state.model;$('generate').disabled=value||!state.model;$('cancel').disabled=!value;
 $('referencePreset').disabled=value;
 updatePatternMode();
 updateDownload(); renderFaceList();
}
function updateDownload(){
 const valid=state.result&&!state.dirty&&!state.busy&&state.result.config.design_version===DESIGN_VERSION&&state.result.display_quality?.watertight;
 $('repairDisplay').disabled=state.busy||!state.result||state.result.display_quality?.version===2;
 for(const id of ['download','saveDesign']){ $(id).classList.toggle('disabled',!valid);$(id).setAttribute('aria-disabled',String(!valid));}
 if(valid){$('download').href=`/artifacts/${state.result.id}/result.stl`;$('saveDesign').href=`/artifacts/${state.result.id}/design.json`;}
 else{$('download').removeAttribute('href');$('saveDesign').removeAttribute('href');}
}
let estimateTimer, estimateSequence=0;
function updatePatternMode(){
 const adaptive=$('surfaceMode').value==='perforated'&&['vertex','aligned','adaptive'].includes($('seamStyle').value);
 const atlas=$('surfaceMode').value==='perforated'&&$('seamStyle').value==='atlas';
 $('surfacePattern').disabled=state.busy||adaptive;
 for(const option of $('surfacePattern').options) option.disabled=atlas&&option.value==='circle';
 $('patternModeHint').hidden=!(adaptive||atlas);
 $('surfaceCellLabel').textContent=adaptive||atlas?'曲面孔距目标尺度':'曲面孔中心距';
 $('patternModeHint').textContent=$('seamStyle').value==='vertex'?'顶点模式仅支持 BCC + 自动连接。跨面共享定向多边形孔网，取消边中部连杆；从多边形顶点向内生成短柱，再以直杆接入内部节点或相邻顶点的内侧节点。局部会改变蜂窝拓扑。':['aligned'].includes($('seamStyle').value)?'定向模式使用跨面规则孔列，不使用上方固定孔型。结构轴向控制孔列方向；曲率变化处允许局部改变孔边数，孔距是目标尺度。':'旧自适应模式不使用固定孔型；孔距表示等面积目标尺度，各孔的边数、方向和实际间距可以变化。';
 if(atlas) $('patternModeHint').textContent='曲面分区铺网：六边形/方形孔型实际参与排孔，沿局部曲面铺设并检查漏铺区域。仅支持 BCC + 自动连接；内部杆只接多边形顶点。曲率变化与分区接合处允许过渡多边形。';
}
function changed(){
 state.dirty=true;updateDownload();
 $('exportHint').textContent=state.result?'参数或面选择已改变，请重新生成后导出。':'STL 导出完整网格；视图可能使用减面副本。';
 const kind=$('latticeType').value, label=$('latticeType').selectedOptions[0].textContent;
 $('structureDesc').textContent=descriptions[kind][0];$('cellHint').textContent=descriptions[kind][1];
 $('structureIcon').innerHTML=polygons[kind]||polygons[kind.startsWith('hex')?'hex':'cubic'];
 $('wallLabel').textContent=struts.has(kind)?'连杆线径':'结构壁厚';
 $('layerField').hidden=!['hexcell','hex3d'].includes(kind);
 const local=$('scope').value==='faces',perforate=$('unselected').value==='perforate';
 $('unselectedField').hidden=!local;$('depthField').hidden=!local||perforate;
 const sheet=$('surfaceMode').value==='perforated';$('surfaceFields').hidden=!sheet;
 updatePatternMode();
 $('scopeHint').textContent=local&&!perforate?'仅所选面的向内距离带参与设计，其余区域保持实心。':sheet?'设计面与未选面均沿原曲面排孔，内接立体晶格；橙色保留面仍为实心层。':'只对体积晶格裁切，没有沿原曲面连续排孔的面层。';
 $('metricType').textContent=label;$('metricSize').textContent=`${$('cell').value} / ${$('wall').value} mm`;
 if(state.result){$('metricDensity').textContent='待更新';$('metricFaces').textContent='待更新';}
 clearTimeout(estimateTimer);estimateTimer=setTimeout(updateEstimate,450);
 remember();
}
async function updateEstimate(){
 if(!state.model||state.busy)return;
 const seq=++estimateSequence;
 try{const e=await api('/api/estimate',request('final'));if(seq!==estimateSequence)return;
  $('estimate').textContent=`设定精度：${(e.voxels/1e6).toFixed(1)} / ${e.limit/1e6} 百万采样点 · ${e.memory_mode==='chunked'?'低内存分块':'整块计算'} · 粗估内存 ${e.memory_gb} GB（非峰值上限）${e.scratch_gb?'\n临时磁盘预留约 '+e.scratch_gb+' GB':''}${e.physical_memory?'\n当前可用物理内存 '+e.physical_memory.available_gb+' GB / 总计 '+e.physical_memory.total_gb+' GB':''}${e.allowed?'':'\n'+e.blocked_reason}${e.warnings.length?'\n'+e.warnings.join('\n'):''}`;
 }catch(err){if(seq===estimateSequence)$('estimate').textContent=err.message;}
}
function setRole(role){state.role=role;for(const r of ['design','preserve']){const btn=$(r==='design'?'roleDesign':'rolePreserve');btn.classList.toggle('active',r===role);btn.setAttribute('aria-pressed',String(r===role));}
 $('roleHint').textContent=role==='design'?'绿色面沿原曲面排孔；默认未选面也开孔。橙色保留面保持实心，若保留内侧面则不会内外贯通。':'橙色面保留指定厚度的实心层，不开孔。若想内外贯通，请取消内侧面的橙色保留标记。';
 setView('source');
}
function toggleFace(fid){
 if(state.busy)return;
 const selected=state[state.role], other=state[state.role==='design'?'preserve':'design'];
 if(selected.has(fid))selected.delete(fid);else{selected.add(fid);other.delete(fid);}
 if(state.role==='preserve'&&selected.size&&Number($('shell').value)===0)$('shell').value='0.8';
 if(state.role==='design'&&selected.size)$('scope').value='faces';
 if(!state.preserve.has(fid))delete state.faceShells[fid];
 refreshSelection();changed();
}
function refreshSelection(){
 $('designCount').textContent=state.design.size;$('preserveCount').textContent=state.preserve.size;
 if(state.source){for(const group of state.source.geometry.groups){const fid=group.faceId;group.materialIndex=state.design.has(fid)?1:state.preserve.has(fid)?2:0;}}
 renderFaceList();
}
function renderFaceList(){
 refreshPreserveEditor();
 if(!state.model)return;
 const filter=$('faceSearch').value.toLowerCase();const list=$('faceList');list.replaceChildren();
 for(const face of state.bodies[state.body].faces){
  if(!`${face.label} ${face.type}`.toLowerCase().includes(filter))continue;
  const btn=document.createElement('button');btn.className='face-item';btn.disabled=state.busy;
  btn.classList.toggle('design',state.design.has(face.id));btn.classList.toggle('preserve',state.preserve.has(face.id));
  btn.setAttribute('aria-pressed',String(state.design.has(face.id)||state.preserve.has(face.id)));
  btn.title=`${face.label} · ${face.type} · ${face.area.toFixed(1)} mm²`;
  const mark=document.createElement('span');mark.className='face-mark';
  const name=document.createElement('span');name.textContent=face.label;
  const detail=document.createElement('span');detail.className='face-meta';detail.textContent=state.preserve.has(face.id)?`${face.type} · ${state.faceShells[face.id]??$('shell').value} mm`:face.type;
  btn.append(mark,name,detail);btn.onclick=()=>{setView('source');toggleFace(face.id);};list.append(btn);
 }
}
function refreshPreserveEditor(){
 const select=$('preserveFace'),previous=select.value;
 state.faceShells=cleanThicknesses(state.faceShells,state.preserve);
 const faces=(state.bodies[state.body]?.faces??[]).filter(face=>state.preserve.has(face.id));
 select.replaceChildren(...faces.map(face=>{const option=document.createElement('option');option.value=face.id;option.textContent=face.label;return option;}));
 if(faces.some(face=>String(face.id)===previous))select.value=previous;
 for(const id of ['preserveFace','faceShell','applyFaceShell','resetFaceShell'])$(id).disabled=state.busy||!faces.length;
 const fid=select.value;
 $('faceShell').value=fid===''?'':state.faceShells[fid]??$('shell').value;
 const custom=Object.entries(state.faceShells).map(([id,t])=>`面 ${String(Number(id)+1).padStart(3,'0')}：${t} mm`);
 $('faceShellSummary').textContent=custom.length?custom.join('；'):'未单独设置，所有保留面使用默认厚度。';
}
function disposeGroup(group){for(const child of [...group.children]){child.traverse(obj=>{obj.geometry?.dispose();if(obj.material)for(const m of Array.isArray(obj.material)?obj.material:[obj.material])m.dispose();});group.remove(child);}}
function resetResult(){disposeGroup(resultRoot);state.result=null;state.resultGroup=null;state.dirty=true;for(const btn of document.querySelectorAll('[data-view]'))btn.disabled=btn.dataset.view!=='source';setView('source');$('metricDensity').textContent='—';$('metricFaces').textContent='—';updateDownload();}
async function showBody(){
 busy(true);
 try{
  const data=await api(`/api/body?model=${state.model}&body=${state.body}`);
  resetResult();disposeGroup(sourceGroup);state.design.clear();state.preserve.clear();state.faceShells={};
  const geom=new THREE.BufferGeometry();geom.setAttribute('position',new THREE.Float32BufferAttribute(data.vertices,3));geom.setIndex(data.indices);geom.computeVertexNormals();
  let start=0;while(start<data.face_ids.length){let end=start+1;while(end<data.face_ids.length&&data.face_ids[end]===data.face_ids[start])end++;geom.addGroup(start*3,(end-start)*3,0);geom.groups.at(-1).faceId=data.face_ids[start];start=end;}
  const mats=[0xc0cec7,0x27a58c,0xe2ac59].map(color=>new THREE.MeshStandardMaterial({color,metalness:.15,roughness:.6,side:THREE.DoubleSide}));
  const mesh=new THREE.Mesh(geom,mats);mesh.userData.faceIds=data.face_ids;sourceGroup.add(mesh);state.source=mesh;
  state.bounds=state.bodies[state.body].bounds;
  state.centre.fromArray(state.bounds[0]).add(new THREE.Vector3().fromArray(state.bounds[1])).multiplyScalar(.5);
  sourceGroup.position.copy(state.centre).negate();resultRoot.position.copy(sourceGroup.position);
  const extent=new THREE.Vector3().fromArray(state.bounds[1]).sub(new THREE.Vector3().fromArray(state.bounds[0]));state.radius=extent.length()/2;
  $('modelSize').textContent=extent.toArray().map(v=>v.toFixed(1)).join(' × ')+' mm';
  if(grid){scene.remove(grid);grid.geometry.dispose();grid.material.dispose();}
  grid=new THREE.GridHelper(state.radius*3,24,0xc9d6ce,0xe1e8e2);grid.rotation.x=Math.PI/2;grid.position.z=-extent.z/2-.5;grid.material.transparent=true;grid.material.opacity=.55;scene.add(grid);
  $('sourceWarning').hidden=state.bodies[state.body].watertight;
  $('sourceWarning').textContent='源网格未闭合。内部判定、密度与尺寸仅供参考；请先检查 CAD 缝隙。';
  $('viewMeshInfo').textContent=`${state.bodies[state.body].faces.length} 张 CAD 面`;
  fit('iso');refreshSelection();changed();updateClip();setView('source');
 }finally{busy(false);updateEstimate();}
}
function fit(kind='iso'){
 const r=state.radius;controls.target.set(0,0,0);camera.near=Math.max(.01,r/1000);camera.far=r*60;
 const direction=kind==='front'?new THREE.Vector3(0,-1,0):kind==='outer'?new THREE.Vector3(-1,1.8,1.1).normalize():new THREE.Vector3(1,-1.8,1.1).normalize();
 const distance=r/Math.sin(THREE.MathUtils.degToRad(camera.fov/2))/Math.min(1,camera.aspect)*1.1;
 camera.position.copy(direction.multiplyScalar(distance));camera.updateProjectionMatrix();controls.update();
}
function setView(view){
 if(view!=='source'&&!state.result)return;
 state.view=view;sourceGroup.visible=view!=='result';resultRoot.visible=view!=='source';
 if(state.source)for(const m of state.source.material){m.transparent=view==='overlay';m.opacity=view==='overlay'?.2:1;m.depthWrite=view!=='overlay';}
 for(const b of document.querySelectorAll('[data-view]'))b.classList.toggle('active',b.dataset.view===view);
 $('viewTitle').textContent=({source:'查看与选择曲面',result:'晶格生成结果',overlay:'原始曲面 · 晶格叠加'})[view];
}
function updateClip(){
 const axis=$('clipAxis').value,index={x:0,y:1,z:2}[axis],ratio=Number($('clipPosition').value)/100;
 $('clipValue').textContent=`${Math.round(ratio*100)}%`;
 if(!state.bounds)return;
 const lo=state.bounds[0][index]-state.centre.getComponent(index),hi=state.bounds[1][index]-state.centre.getComponent(index);
 clipPlane.normal.set(0,0,0).setComponent(index,-1);clipPlane.constant=lo+(hi-lo)*ratio;
 renderer.clippingPlanes=$('clipEnabled').checked?[clipPlane]:[];
}
const raycaster=new THREE.Raycaster();let down;
renderer.domElement.addEventListener('pointerdown',e=>{down={x:e.clientX,y:e.clientY,button:e.button};});
renderer.domElement.addEventListener('pointerup',e=>{
 if(!down||down.button!==0||state.busy||!state.source||!sourceGroup.visible||Math.hypot(e.clientX-down.x,e.clientY-down.y)>4)return;
 const rect=renderer.domElement.getBoundingClientRect();const mouse=new THREE.Vector2((e.clientX-rect.left)/rect.width*2-1,-(e.clientY-rect.top)/rect.height*2+1);raycaster.setFromCamera(mouse,camera);
 const hit=raycaster.intersectObject(state.source).find(h=>!$('clipEnabled').checked||clipPlane.distanceToPoint(h.point)>=0);
 if(hit)toggleFace(state.source.userData.faceIds[hit.faceIndex]);
 down=null;
});

async function waitJob(id,onDone){
 state.job=id;
 while(state.job===id){
  let data;try{data=await api(`/api/status?job=${id}`);}catch(err){state.job=null;busy(false);message('与计算服务失去连接。'+err.message,true);return;}
  $('jobTitle').textContent=data.message;$('jobPercent').textContent=`${data.progress}%`;$('jobProgress').style.width=`${data.progress}%`;
  if(data.memory)$('memoryStatus').textContent=`计算进程当前内存 ${data.memory.rss_gb.toFixed(2)} GB · 峰值 ${data.memory.peak_rss_gb.toFixed(2)} GB`;
  if(data.state==='done'){
   state.job=null;
   try{await onDone(data.result);}catch(err){message(err.message,true);}finally{busy(false);}
   return;
  }
  if(['failed','cancelled'].includes(data.state)){
   state.job=null;busy(false);message(data.message,data.state==='failed');return;
  }
  await new Promise(resolve=>setTimeout(resolve,700));
 }
}
async function loadModel(data,file){
 if(state.busy)return;
 busy(true);message('正在读取 STEP 曲面与实体…');$('importDialog').close();
 try{
  const response=await api(file?'/api/upload':'/api/load',file||data,!!file);
  await waitJob(response.job,async result=>{
   state.model=response.job;state.bodies=result.bodies;state.body=0;
   $('filename').textContent=file?.name||result.filename;$('filename').title=$('filename').textContent;
   $('bodySelect').replaceChildren(...result.bodies.map(b=>{const opt=document.createElement('option');opt.value=b.id;opt.textContent=b.name;return opt;}));
   await showBody();message(`已载入 ${result.bodies.length} 个实体。点击曲面或左侧面列表开始设计。`);
  });
 }catch(err){busy(false);message(err.message,true);}
}
async function showResult(id,result){
   disposeGroup(resultRoot);
   state.result={id,...result};
   if(result.display_quality?.version!==2||!result.display_quality.watertight){
    state.dirty=true;setView('source');updateDownload();
    for(const b of document.querySelectorAll('[data-view]'))b.disabled=b.dataset.view!=='source';
    message('旧显示网格未经过流形检查，已停止显示以免误判碎面。点击“修复旧结果显示”可重建显示副本；原 STL 不变。',true);return;
   }
   const gltf=await new GLTFLoader().loadAsync(`/artifacts/${id}/display.glb`);
   const wire=$('wireToggle').getAttribute('aria-pressed')==='true';
   gltf.scene.traverse(obj=>{if(obj.isMesh){obj.geometry.computeVertexNormals();obj.material.dispose();obj.material=new THREE.MeshStandardMaterial({color:0x548d80,metalness:.27,roughness:.47,side:THREE.DoubleSide,wireframe:wire});}});
   resultRoot.add(gltf.scene);state.resultGroup=gltf.scene;state.dirty=result.config.design_version!==DESIGN_VERSION;
   for(const b of document.querySelectorAll('[data-view]'))b.disabled=false;
   setView('result');
   const density=result.stats.mesh_density??result.stats.density;
   $('densityLabel').textContent=result.stats.mesh_density!=null?'网格体积比':'体素估算密度';
   $('metricDensity').textContent=`${(density*100).toFixed(1)}%`;$('metricFaces').textContent=(result.stats.faces/10000).toFixed(1)+' 万';
   $('viewMeshInfo').textContent=`显示 ${(result.display_faces/10000).toFixed(1)} 万面 · 完整 ${(result.stats.faces/10000).toFixed(1)} 万面`;
   $('exportHint').textContent=`${result.config.quality==='preview'?'预览结果 · 请勿作为制造定稿。':'设定精度结果 · 制造前仍须验证。'} STL ${(result.bytes/1e6).toFixed(1)} MB，采样 ${result.stats.pitch.toFixed(3)} mm。`;
   const perf=result.performance;$('memoryStatus').textContent=perf?.memory?`${perf.scope==='postprocess_only'?'检查点后处理':'本次计算'} ${(perf.elapsed_seconds/60).toFixed(1)} 分钟 · 进程峰值内存 ${perf.memory.peak_rss_gb.toFixed(2)} GB${result.stats.memory_mode==='chunked'?' · 低内存分块':''}`:'';
   const continuity=result.stats.surface_network==='vertex_aligned_rows'?`顶点专用连接：${result.stats.vertex_connected_nodes} / ${result.stats.polygon_vertices} 个多边形顶点接入内部，非顶点落点 ${result.stats.nonvertex_terminals} 个。已替换 ${result.stats.removed_old_outer_members} 根旧外层杆（含 ${result.stats.grazing_members_removed} 根擦边杆）。${result.stats.vertex_posts_unconnected} 个顶点未加内接柱；不会生成悬空短柱。采用顶点短柱与内侧直杆，转折位于面层下方。`:result.stats.surface_network==='aligned_surface_rows'?`定向规则孔列 ${result.stats.aligned_rows} 行 / ${result.stats.adaptive_sites} 点，跨 CAD 面共享方向和相位。${result.stats.continuity_mode==='aligned_straight_bcc'?`${result.stats.routed_terminals} 个杆端以直杆接入；转接角中位数 ${result.stats.straight_angle_deg?.[1]?.toFixed(1)??'—'}°，P95 ${result.stats.straight_angle_deg?.[2]?.toFixed(1)??'—'}°。局部省略 ${result.stats.omitted_boundary_members} / ${result.stats.boundary_members_considered} 根无法安全匹配的边界杆。`:'当前内部结构未启用 BCC 直杆匹配。'}`:result.stats.surface_network==='shared_surface_voronoi'?`统一曲面取点 ${result.stats.adaptive_sites} 处，跨面共享不规则孔网；${result.stats.routed_terminals??0} 个 BCC 杆端接入孔边。不使用逐面固定孔型或接缝补线。`:result.stats.continuity_mode==='boundary_bcc'?`已转接 ${result.stats.routed_terminals} 个 BCC 杆端；${result.stats.seam_style==='stitch'?`曲面取点连接 ${result.stats.surface_core_connections} 处，无分界直筋`:`曲面接缝连筋 ${(result.stats.seam_rail_length_mm??0).toFixed(1)} mm`}。局部改变连接结构，保留主体单元与杆径。`:'';
   const atlasInfo=result.stats.surface_network==='atlas_vertex_rows'?`曲面分区铺网：${result.stats.adaptive_sites} 个孔中心；覆盖检查 P95 ${result.stats.atlas_coverage_p95_mm.toFixed(2)} mm，超目标孔距区域 ${(result.stats.atlas_uncovered_fraction*100).toFixed(2)}%。${result.stats.vertex_connected_nodes} / ${result.stats.polygon_vertices} 个多边形顶点接入内部，非顶点落点 ${result.stats.nonvertex_terminals} 个。分区与曲率过渡处允许改变孔边数。`:'';
   message([`显示网格已通过流形检查（容差 ${result.display_quality.tolerance_mm.toFixed(3)} mm）。完整模型${result.stats.watertight?'闭合':'未闭合'}；不代表已验证强度或可打印性。`,atlasInfo,continuity,...result.warnings].filter(Boolean).join('\n'),result.warnings.length>0);
   remember();
}
async function generate(quality){
 if(state.busy||!state.model)return;
 busy(true);$('jobTitle').textContent='检查参数';$('jobPercent').textContent='0%';$('jobProgress').style.width='0%';$('memoryStatus').textContent='';message(quality==='preview'?'快速预览会根据模型大小自动降低精度。':'正在按设定精度生成，期间可随时停止。');
 try{
  const payload=request(quality),e=await api('/api/estimate',payload);
  if(!e.allowed)throw new Error(e.blocked_reason||'本次配置未通过采样检查');
  if(e.memory_risk||e.voxels>80000000){
   const ram=e.physical_memory?`当前可用物理内存 ${e.physical_memory.available_gb} GB。`:'未能读取可用物理内存。';
   if(!confirm(`本次约 ${(e.voxels/1e6).toFixed(1)} 百万采样点，粗估内存 ${e.memory_gb} GB；${ram}\n网格和布尔运算可能需要更多内存，导致卡顿或生成失败。确定开始？`)){busy(false);message('已取消启动高内存任务，未开始生成。');return;}
  }
  const response=await api('/api/generate',payload);
  await waitJob(response.job,result=>showResult(response.job,result));
 }catch(err){busy(false);$('jobTitle').textContent='未启动生成';message(err.message,true);}
}
for(const id of settings)$(id).addEventListener('input',changed);
$('shell').addEventListener('input',renderFaceList);
$('preserveFace').onchange=()=>{$('faceShell').value=state.faceShells[$('preserveFace').value]??$('shell').value;};
$('applyFaceShell').onclick=()=>{
 if(state.busy)return;
 try{state.faceShells=setThickness(state.faceShells,state.preserve,Number($('preserveFace').value),$('faceShell').value);refreshSelection();changed();}
 catch(err){message(err.message,true);}
};
$('resetFaceShell').onclick=()=>{if(state.busy)return;delete state.faceShells[$('preserveFace').value];refreshSelection();changed();};
$('referencePreset').onclick=()=>{
 if(state.busy)return;
 const preset={surfaceMode:'perforated',surfacePattern:'hex',surfaceThickness:.8,surfaceCell:6.4,surfaceWidth:.7,poreRounding:.35,junctionBlend:.25,edgeWidth:1,surfaceContinuity:'auto',seamStyle:'vertex',latticeType:'bcc',cell:7.3,wall:1.2,axis:'z',pitch:.2,smooth:2};
 for(const [id,value] of Object.entries(preset))$(id).value=value;
 changed();message('已应用右肩参考件近似参数：曲面孔距 6.4 / 筋宽 0.7 mm，内部 BCC 周期 7.3 / 杆径 1.2 mm。原面选择未变，尚未生成；橙色保留面仍实心。');
};
$('roleDesign').onclick=()=>setRole('design');$('rolePreserve').onclick=()=>setRole('preserve');
$('clearFaces').onclick=()=>{state.design.clear();state.preserve.clear();state.faceShells={};refreshSelection();changed();};
$('faceSearch').oninput=renderFaceList;
$('bodySelect').onchange=async()=>{state.body=Number($('bodySelect').value);try{await showBody();}catch(err){message(err.message,true);}};
for(const b of document.querySelectorAll('[data-view]'))b.onclick=()=>setView(b.dataset.view);
$('fitView').onclick=()=>fit();$('isoView').onclick=()=>fit('iso');$('frontView').onclick=()=>fit('front');
$('wireToggle').onclick=()=>{const active=$('wireToggle').getAttribute('aria-pressed')!=='true';$('wireToggle').setAttribute('aria-pressed',String(active));$('wireToggle').classList.toggle('active',active);for(const root of [sourceGroup,resultRoot])root.traverse(obj=>{if(obj.isMesh)for(const m of Array.isArray(obj.material)?obj.material:[obj.material])m.wireframe=active;});};
for(const id of ['clipEnabled','clipAxis','clipPosition'])$(id).oninput=updateClip;
$('preview').onclick=()=>generate('preview');$('generate').onclick=()=>generate('final');
$('repairDisplay').onclick=async()=>{
 if(state.busy||!state.result)return;
 const id=state.result.id,wasDirty=state.dirty;busy(true);message('只重建显示副本，原 STL 和原任务均保留。');
 try{const response=await api('/api/redisplay',{job:id});await waitJob(response.job,async result=>{await showResult(response.job,result);if(wasDirty)changed();remember();});}
 catch(err){busy(false);message(err.message,true);}
};
$('cancel').onclick=async()=>{if(!state.job)return;try{await api('/api/cancel',{job:state.job});}catch(err){message(err.message,true);}};
$('shutdown').onclick=async()=>{if(!confirm('关闭本地软件服务？正在进行的生成也会停止；已有输出会保留。'))return;try{await api('/api/shutdown',{});state.job=null;busy(true);$('cancel').disabled=true;$('shutdown').disabled=true;message('软件服务已关闭。下次双击启动文件即可重新打开。');}catch(err){message(err.message,true);}};
$('openImport').onclick=()=>$('importDialog').showModal();$('demo').onclick=()=>loadModel({demo:true});
$('loadPath').onclick=()=>loadModel({path:$('filePath').value.trim()});
function loadFile(file){if(!file)return;if(!/\.(step|stp)$/i.test(file.name)){message('请选择 .step 或 .stp 文件',true);return;}loadModel(null,file);}
$('fileInput').onchange=()=>loadFile($('fileInput').files[0]);
$('dropzone').ondragover=e=>{e.preventDefault();$('dropzone').classList.add('dragover');};
$('dropzone').ondragleave=()=>$('dropzone').classList.remove('dragover');
$('dropzone').ondrop=e=>{e.preventDefault();$('dropzone').classList.remove('dragover');loadFile(e.dataTransfer.files[0]);};
changed();
try{
 const bootstrap=await api('/api/bootstrap');state.token=bootstrap.token;
 $('sampleLimits').textContent=bootstrap.sample_limits?`数量上限：快速预览 ${bootstrap.sample_limits.preview/1e4} 万点；精细生成 ${bootstrap.sample_limits.final/1e8} 亿点。提高上限不等于内存足够，也不会修复排孔映射或显示减面问题。`:'服务尚未更新，请重启软件以启用新的采样数量上限。';
 let saved;try{saved=JSON.parse(localStorage.getItem('latticeStudio.session'));}catch{}
 if(reviewJob){
  const reviewed=await api(`/api/status?job=${encodeURIComponent(reviewJob)}`);
  if(reviewed.state!=='done'||!reviewed.result.config)throw new Error('该结果尚未完成或不存在');
  const r=reviewed.result,c=r.config;
  saved={version:c.design_version,model:r.model_id,body:r.source_body,filename:'开孔曲面结果检查',
   settings:{surfaceMode:c.surface_mode,surfacePattern:c.surface_pattern,surfaceThickness:c.surface_thickness,surfaceCell:c.surface_cell??0,surfaceWidth:c.surface_width??0,poreRounding:c.pore_rounding??0,junctionBlend:c.junction_blend??0,edgeWidth:c.edge_width??0,surfaceContinuity:c.surface_continuity??'auto',seamStyle:c.seam_style??'stitch',latticeType:c.lattice,cell:c.cell,wall:c.wall,axis:c.axis,layer:c.layer,scope:c.scope,unselected:c.unselected,depth:c.depth,shell:c.shell,pitch:c.pitch,smooth:c.smooth,memoryMode:c.memory_mode??'auto'},
   design:c.design_faces,preserve:c.preserve_faces,faceShells:c.preserve_thicknesses??{},result:reviewJob,dirty:false};
 }
 if(saved?.model){
  state.restoring=true;
  try{
   const loaded=await api(`/api/status?job=${saved.model}`);
   if(loaded.state!=='done'||!loaded.result.bodies)throw new Error('上次的模型已不可用');
   state.model=saved.model;state.bodies=loaded.result.bodies;state.body=Number(saved.body);
   $('filename').textContent=saved.filename;
   $('bodySelect').replaceChildren(...state.bodies.map(b=>{const option=document.createElement('option');option.value=b.id;option.textContent=b.name;return option;}));
   $('bodySelect').value=state.body;
   await showBody();
   for(const id of settings)if(saved.settings[id]!==undefined)$(id).value=saved.settings[id];
   state.design=new Set(saved.design);state.preserve=new Set(saved.preserve);state.faceShells=cleanThicknesses(saved.faceShells,state.preserve);refreshSelection();changed();
   let outdated=saved.version!==DESIGN_VERSION;
   if(saved.result){const previous=await api(`/api/status?job=${saved.result}`);if(previous.state==='done'){outdated=outdated||previous.result.config.design_version!==DESIGN_VERSION;await showResult(saved.result,previous.result);if(saved.dirty||outdated)changed();}}
   if(outdated)changed();
   if(outdated)message(state.result?.display_quality?.version===2?'已载入旧版结果，原 STL 未改变；请重新生成以应用当前算法。右侧可选择参考件风格，独立调整曲面孔距、筋宽、圆角和实边。':'碎面保护已更新：旧结果需重新生成以应用排孔修复。旧显示副本可单独修复；未经检查的旧减面网格不再展示。',true);
   else if(!state.result)message('已恢复上次模型与参数。');
   else{$('jobTitle').textContent='已载入完成结果';$('jobPercent').textContent='100%';$('jobProgress').style.width='100%';}
   if(reviewJob)fit('outer');
  }catch(err){state.model=null;message(err.message,true);await loadModel({demo:true});}
  finally{state.restoring=false;busy(false);remember();updateEstimate();}
 }else{await loadModel({demo:true});}
}catch(err){message('启动失败：'+err.message,true);}
