# Sprite style transfer

Fast neural style transfer of RGBA sprite images. 
Transform the style of a sprite sheet from one platform to another!

![NES to SNES sprite conversion example](/assets/mm9_example_output.png?cachebust=1)

## Getting started

### Installation
Install the dependencies using [pipenv](https://github.com/pypa/pipenv)
```
pipenv install
```

### Source images
Place your sprite sheet in the images/inputs folder. These should be RGBA pngs with an alpha channel.

### Running
```
pipenv run style_image --model_path ./data/models/megaman8.pth --image_path ./images/inputs/your_image.png
```

Your image will be saved in images/outputs/<model_name>/<image_filename>

### Cleanup

The engine outputs full RGBA images, but real super nintendo sprites have a color depth of [15 colors per layer](https://en.wikipedia.org/wiki/List_of_video_game_console_palettes#Super_NES_(SNES)). You can make your outputted sprite sheets look more plausible by reducing the color depth to 15 or 30.


![Mummy sprite color reduction example](/assets/mummy_cleanup.png?cachebust=1)

From left to right: Original image, raw rgb output, reduced to 15 colors, manual edit of eyes and hand

### Styles

Different models will output differently styled sprites. It helps if the model was trained on images in a similar genre as your source image, though this is not required. Experiment with a few different style models to see what works best for the look you're trying to achieve


![Style examples](/assets/style_examples.png?cachebust=1)

From top to bottom: Original image, styled using a model trained on Megaman 8, styled using a model trained on Castlevania SOTN

## References
[Fast Neural Style Transfer in PyTorch](https://github.com/eriklindernoren/Fast-Neural-Style-Transfer) (I used this project as a starting point)
