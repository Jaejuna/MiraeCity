## data convert thread
import pandas as pd

data = [
 { "go" : { 'ɑpʰɯɾo' : 0.80, 'ɑpʰɯɾo kɑ' : 1.0 } },
 { "stop" : { 'mʌmtɕʰwʌ' : 0.8, 'kɯmɑn' : 0.85, 'sɯtʰɑp' : 0.90 } },
 { "back" : { 'toɾɑɡɑ' : 0.8, 'twiɾo kɑ' : 0.9, 'p*ɑɡu' : 0.95 } },
 { "goback" : { 'pokk*wi' : 0.9, 'wʌnwitɕʰi' : 0.85 } }
]

# 데이터 파싱을 위한 리스트
commands = []
pronunciations = []
similarities = []

for item in data:
    for command, pronunciation_dict in item.items():
        for pronunciation, similarity in pronunciation_dict.items():
            commands.append(command)
            pronunciations.append(pronunciation)
            similarities.append(similarity)

# DataFrame 생성
df = pd.DataFrame({
    'Command': commands,
    'Pronunciation': pronunciations,
    'Similarity': similarities
})

print(df)
