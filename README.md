[English/[Japanese](README_JP.md)]

# Ninja Lantern

![Ninja Lantern](images/ninja-lantern.png)

https://github.com/karaage0703/ninja-lantern/assets/5562157/0f9bfaca-e2e7-49ec-abc8-1d2eb3cc0807

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
$ cd && git clone https://github.com/karaage0703/ninja-lantern.git
```

4. Move into the `ninja-lantern` directory.

```bash
$ cd ninja-lantern
```

5. Download and copy AI model and utility files of `NARUTO-HandSignDetection` 

```bash
$ cd && git clone https://github.com/Kazuhito00/NARUTO-HandSignDetection
$ cd ~/ninja-lantern
$ cp -r ~/NARUTO-HandSignDetection/model ./
$ cp -r ~/NARUTO-HandSignDetection/utils ./
$ cp -r ~/NARUTO-HandSignDetection/setting ./
```

6. Run the script to start the Ninja Lantern.

Single process version

```bash
$ sudo python3 ninja_lantern.py
```

Multi process version and full screen

```bash
$ sudo python3 ninja_lantern_mp.py --full_screen True
```

## Usage

Simply bring your hand seal in front of the camera and enjoy as the lamp changes color accordingly.

## Contributions

Feedback and pull requests are always welcome.

## License

[MIT License](LICENSE)

## Special Thanks

- [Kazuhito00](https://github.com/Kazuhito00)
- [PINTO0309](https://github.com/PINTO0309)

# References

- [Kazuhito00/NARUTO-HandSignDetection](https://github.com/Kazuhito00/NARUTO-HandSignDetection)
- [rpi-ws281x/rpi-ws281x-python](https://github.com/rpi-ws281x/rpi-ws281x-python)
