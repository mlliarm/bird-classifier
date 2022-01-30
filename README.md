
________________________________________________________________________
MIT License

Copyright (c) 2022 Michail Liarmakopoulos <mlliarm@yandex.com>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
________________________________________________________________________

# birdclassifier

This app is a CLI bird classifier using a specific pretrained [Tensorflow model](https://tfhub.dev/google/aiy/vision/classifier/birds_V1/1) and specific [bird labels](https://www.gstatic.com/aihub/tfhub/labelmaps/aiy_birds_V1_labelmap.csv).

As it is now, it needs network access so that it works, since both the model and the labels are being fetched from the internet.

## Installation

Simply run within a python virtual environment running Python of version `3.9` or greater:

```python3
pip install -r requirements.txt
```

## Use

This app comes with a simple but handy CLI API.

We've put some test images of birds in the images directory, `imgs/`.

To test some of them simply run from within this directory:

```python
python runner.py file://$PWD/imgs/bird_01.jpg
```

or using a url where a bird image has been stored:

```python
python runner.py 'https://i.imgur.com/8eGMhGP.jpg'
```

or with more than one image:

```python
python runner.py 'https://i.imgur.com/8eGMhGP.jpg', 
'https://i.imgur.com/TRVxZAZ.jpg',
'https://i.imgur.com/kBHq8Xt.jpg',
'https://i.imgur.com/wmEaY0t.jpg',
'https://i.imgur.com/olSQAGI.jpg'
```

If you try running the `runner.py` without inserting any image, or comma separated images, you should get a warning.

To get help on how to run the `runner.py` script, write:

```python
python runner.py --help
```

## Supported versions of CPython
- 3.9
- 3.10

## Supported OSs:
- Ubuntu 18.04, 20.04 (ubuntu-latest).
- MacOSX 10.15, 11.6.2 (macos-latest).
- Windows 10.

## Known issues
- Currently the library fails to work properly with python 3.6, 3.7, 3.8.
- CI breaks to run on Windows Server (2016, 2019, 2022).
