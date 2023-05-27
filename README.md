[English/[Japanese](README_JP.md)]

# Ninja Lantern

![Ninja Lantern](images/ninja-lantern.jpg)

The Ninja Lantern is an interactive Turkish lamp that changes colors in response to hand seals, just like those used by ninjas. It uses Python and machine learning to recognize the seals and adjust the lamp's color accordingly.

## Features

- Real-time recognition of ninja hand seals using a camera
- Displays the color corresponding to the hand seal using an RGB LED
- Infinite possibilities and combinations of colors

## Required Hardware

- Raspberry Pi (with a USB Camera)
- Turkish lamp
- RGB LED(Neopixel)
- Appropriate wiring

## Setup

1. Gather the necessary hardware and connect it to the Raspberry Pi.
2. Install the necessary Python libraries on the Raspberry Pi's OS.

Setup Edge AI CV libraries

```bash
$ sudo apt-get update
$ cd && git clone https://github.com/karaage0703/edge-ai-cv
$ cd ~/edge-ai-cv/setup
$ ./install_opencv.sh
$ sudo pip3 install onnxruntime
```

Setup LED control libraries

```
$ sudo apt-get install scons
$ cd && git clone https://github.com/jgarff/rpi_ws281x.git
$ cd rpi_ws281x
$ sudo scons
$ sudo pip3 install rpi_ws281x
```

3. Clone or download this repository.

```bash
$ git clone https://github.com/username/ninja-lantern.git
```

4. Move into the `ninja-lantern` directory.

```bash
$ cd ninja-lantern
```

5. Download and copy 'NARUTO-HandSignDetection'

```bash
$ cd && git clone https://github.com/Kazuhito00/NARUTO-HandSignDetection
$ cd ~/ninja-lantern
$ cp -r ~/NARUTO-HandSignDetection/model ./
$ cp -r ~/NARUTO-HandSignDetection/utils ./
$ cp -r ~/NARUTO-HandSignDetection/setting ./
```

6. Run the script to start the Ninja Lantern.

```bash
$ python3 main.py
```

## Usage

Simply bring your hand seal in front of the camera and enjoy as the lamp changes color accordingly. The mapping between seals and colors can be customized in the `config.py` file.

## Contributions

Feedback and pull requests are always welcome.

## License

[MIT License](LICENSE)
