from papermage.recipes import CoreRecipe

import json

core_recipe = CoreRecipe()

doc = core_recipe.run("Основные правила оформления чертежей (Хотина, Ермакова, Кожухова)_rotated.pdf")

with open('filename.json', 'w') as f_out:
    json.dump(doc.to_json(), f_out, indent=4)

