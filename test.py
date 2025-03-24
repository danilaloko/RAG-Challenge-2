from papermage.recipes import CoreRecipe

core_recipe = CoreRecipe()

doc = core_recipe.run("Основные правила оформления чертежей (Хотина, Ермакова, Кожухова)_rotated.pdf")

print(doc)