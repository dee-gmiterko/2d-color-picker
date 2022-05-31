import DataExtractor
import ModelConstructor
import ColorPaletteGenerator

PATH = '/home/ienze/Downloads/generative-design/transformation-2d-palette/1/'
DATA_LIMIT = 8000
REMOVE_DUPLICITE_COLORS = True
DUPLICITY_ACCURACY = 4
MODEL_ITERATIONS = 8
MAGIC_SIZE = 42

data = DataExtractor.extractData(PATH, DATA_LIMIT, REMOVE_DUPLICITE_COLORS, DUPLICITY_ACCURACY)
model = ModelConstructor.constructModel(data, MODEL_ITERATIONS, MAGIC_SIZE)
ColorPaletteGenerator.generatePalette(data, model)
