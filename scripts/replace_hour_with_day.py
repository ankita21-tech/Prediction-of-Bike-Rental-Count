import json
p='Rental_bikes_count_Prediction_Model_Comparison.ipynb'
with open(p,'r',encoding='utf-8') as f:
    j=json.load(f)
repl_count=0
for cell in j.get('cells',[]):
    if 'source' in cell:
        new_source=[]
        changed=False
        for line in cell['source']:
            if 'hour.csv' in line:
                line=line.replace('hour.csv','day.csv')
                changed=True
                repl_count+=1
            new_source.append(line)
        if changed:
            cell['source']=new_source
with open(p,'w',encoding='utf-8') as f:
    json.dump(j,f,ensure_ascii=False,indent=1)
print('replacements=',repl_count)
