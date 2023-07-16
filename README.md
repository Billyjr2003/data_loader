# data_loader
A simple way to get your data ready by visualizing and giving labels to it . Then we use the data loader to read the data and send it to the model .

## What do i use to install the data ?
So i used TextRecognitionDataGenerator . It is A synthetic data generator for text recognition .

## What is it for? 
Generating text image samples to train an OCR software. Now supporting non-latin text! For a more thorough tutorial see [the official documentation](https://textrecognitiondatagenerator.readthedocs.io/en/latest/index.html).

## What do I need to make it work?
Install the pypi package

```
pip install trdg
```

Afterwards, you can use `trdg` from the CLI. I recommend using a virtualenv instead of installing with `sudo`.

If you want to add another language, you can clone the repository instead. Simply run `pip install -r requirements.txt`

## Docker image

If you would rather not have to install anything to use TextRecognitionDataGenerator, you can pull the docker image.

```
docker pull belval/trdg:latest

docker run -v /output/path/:/app/out/ -t belval/trdg:latest trdg [args]
```

The path (`/output/path/`) must be absolute.

## New
- Add `--stroke_width` argument to set the width of the text stroke (Thank you [@SunHaozhe](https://github.com/SunHaozhe))
- Add `--stroke_fill` argument to set the color of the text contour if stroke > 0 (Thank you [@SunHaozhe](https://github.com/SunHaozhe))
- Add `--word_split` argument to split on word instead of per-character. This is useful for ligature-based languages
- Add `--dict` argument to specify a custom dictionary (Thank you [@luh0907](https://github.com/luh0907))
- Add `--font_dir` argument to specify the fonts to use
- Add `--output_mask` to output character-level mask for each image
- Add `--character_spacing` to control space between characters (in pixels)
- Add python module
- Add `--font` to use only one font for all the generated images (Thank you [@JulienCoutault](https://github.com/JulienCoutault)!)
- Add `--fit` and `--margins` for finer layout control
- Change the text orientation using the `-or` parameter
- Specify text color range using `-tc '#000000,#FFFFFF'`, please note that the quotes are **necessary**
- Add support for Simplified and Traditional Chinese


## How does it work?

Words will be randomly chosen from a dictionary of a specific language. Then an image of those words will be generated by using font, background, and modifications (skewing, blurring, etc.) as specified.

### Basic (Python module)

The usage as a Python module is very similar to the CLI, but it is more flexible if you want to include it directly in your training pipeline, and will consume less space and memory. There are 4 generators that can be used.

```py
from trdg.generators import (
    GeneratorFromDict,
    GeneratorFromRandom,
    GeneratorFromStrings,
    GeneratorFromWikipedia,
)

# The generators use the same arguments as the CLI, only as parameters
generator = GeneratorFromStrings(
    ['Test1', 'Test2', 'Test3'],
    blur=2,
    random_blur=True
)

for img, lbl in generator:
    # Do something with the pillow images here.
```

