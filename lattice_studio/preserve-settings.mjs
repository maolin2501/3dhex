// JSON/session boundary: only current retained faces can own local thickness.
export function cleanThicknesses(values,preserved){
 const result={};
 for(const [key,value] of Object.entries(values??{})){
  const id=Number(key),thickness=Number(value);
  if(String(id)===key&&Number.isInteger(id)&&preserved.has(id)&&value!==null&&value!==''&&
     typeof value!=='boolean'&&Number.isFinite(thickness)&&thickness>=.1&&thickness<=50)result[key]=thickness;
 }
 return result;
}
export function setThickness(values,preserved,id,value){
 const thickness=Number(value);
 if(!preserved.has(id)||value===''||value===null||typeof value==='boolean'||
    !Number.isFinite(thickness)||thickness<.1||thickness>50)throw new Error('请选择保留面，并输入 0.1–50 mm 的厚度');
 return {...cleanThicknesses(values,preserved),[id]:thickness};
}
