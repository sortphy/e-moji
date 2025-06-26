# e-moji
neymoji v2


## para rodar
- faz venv
- baixa os requirements
- roda o ollama
- baixa um dos modelos do ollama e configura no python qual vai usar

modelo pesado:   
ollama pull qwen2.5vl:7b

modelo leve:   
ollama pull qwen2.5vl:3b




## problemas:
- o ollama so suporta GPU no windows 11 e eu me recuso a atualizar
- entao so da pra rodar na CPU
- fica super travado na CPU
- sexta se o icaro for pra aula a gente testa no notebook dele, nao sei se tem GPU integrada ou standalone, se tb ficar travado a gent em uda pra um modelo mais leve
- vou mudar umas configs pra ver se consigo fazer rodar melhor