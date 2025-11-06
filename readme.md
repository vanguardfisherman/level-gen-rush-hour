comando para activar el script
dir /b *.py
para correrlo
python ".\level_gen.py" > easy.out.ts





para el funcionamiento de nuestro script generador de niveles para rush hour lo hemos dividido en 3 fases
genera niveles apartir de la dificultad que tu elijas

easy (10 niveles preset)
python ".\level_gen.py" --difficulty easy --n-levels 10 > easy.out.ts
python genlevels.py --difficulty easy --n-levels 10 > easy.out.ts

normal con semilla fija y 12 niveles:
python genlevels.py --difficulty normal --n-levels 12 --seed 777 > normal.out.ts

hard pero forzando más piezas y más verticales:
python genlevels.py --difficulty hard --min-pieces 12 --max-pieces 15 --dir-bias 0.7,1.3 > hard.out.ts


override del rango de movimientos para easy:

python genlevels.py --difficulty easy --min-moves 12 --max-moves 18 > easy18.out.ts
