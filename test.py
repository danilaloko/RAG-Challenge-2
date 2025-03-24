from papermage.recipes import CoreRecipe

import json

core_recipe = CoreRecipe()

doc = core_recipe.run("Основные правила оформления чертежей (Хотина, Ермакова, Кожухова)_rotated.pdf")

print(doc.text)

# Сохранить структурированные данные
output_data = {
    "metadata": doc.metadata.to_dict(),
    "text": doc.text,
    "layers": {name: [ent.to_dict() for ent in entities] 
               for name, entities in doc.layers.items()}
}

with open("document_output.json", "w", encoding="utf-8") as f:
    json.dump(output_data, f, ensure_ascii=False, indent=2)
