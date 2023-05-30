#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import csv
import time
import copy
from collections import deque

import cv2 as cv
import numpy as np

from utils import CvFpsCalc
from utils import CvDrawText
from model.yolox.yolox_onnx import YoloxONNX

from rpi_ws281x import PixelStrip, Color
from multiprocessing import Value, Process


# LED strip configuration:
LED_COUNT = 57        # Number of LED pixels.
LED_PIN = 21          # GPIO pin connected to the pixels (21 uses PWM!).
LED_FREQ_HZ = 800000  # LED signal frequency in hertz (usually 800khz)
LED_DMA = 10          # DMA channel to use for generating signal (try 10)
LED_BRIGHTNESS = 255  # Set to 0 for darkest and 255 for brightest
LED_INVERT = False    # True to invert the signal (when using NPN transistor level shift)
LED_CHANNEL = 0       # set to '1' for GPIOs 13, 19, 41, 45 or 53


def display_color_pattern(seal_type):
    # Create NeoPixel object with appropriate configuration.
    strip = PixelStrip(LED_COUNT, LED_PIN, LED_FREQ_HZ, LED_DMA, LED_INVERT, LED_BRIGHTNESS, LED_CHANNEL)
    # Intialize the library (must be called once before other functions).
    strip.begin()
 
    count = 0
    while True:
        count += 1
        if count > 10000:
            count = 0

        if seal_type.value == -1:
            clear_color(strip)
            break
        else:
            if seal_type.value == 0:
                seal = 'seal18'
            if seal_type.value == 1:
                seal = 'seal1'
            if seal_type.value == 2:
                seal = 'seal2'
            if seal_type.value == 3:
                seal = 'seal3'
            if seal_type.value == 4:
                seal = 'seal4'
            if seal_type.value == 5:
                seal = 'seal5'
            if seal_type.value == 6:
                seal = 'seal6'
            if seal_type.value == 7:
                seal = 'seal7'
            if seal_type.value == 8:
                seal = 'seal8'
            if seal_type.value == 9:
                seal = 'seal9'
            if seal_type.value == 10:
                seal = 'seal10'
            if seal_type.value == 11:
                seal = 'seal11'
            if seal_type.value == 12:
                seal = 'seal12'
            if seal_type.value == 13:
                seal = 'seal13'
            if seal_type.value == 14:
                seal = 'seal14'

            """Display LED pattern"""
            seal_effects = {
                "seal1": {"colors": [(255, 0, 0), (0, 255, 0), (0, 0, 255)], "effect": "static"},
                "seal2": {"colors": [(255, 255, 0), (0, 255, 255), (255, 0, 255)], "effect": "blink", "speed": 0.5},
                "seal3": {"colors": [(255, 0, 127), (127, 0, 255), (0, 255, 127)], "effect": "gradient", "speed": 0.5},
                "seal4": {"colors": [(255, 255, 255), (127, 127, 127), (0, 0, 0)], "effect": "blink", "speed": 1},
                "seal5": {"colors": [(255, 0, 255), (0, 255, 255), (255, 255, 0)], "effect": "gradient", "speed": 1},
                "seal6": {"colors": [(255, 127, 0), (127, 255, 0), (0, 255, 127)], "effect": "blink", "speed": 0.5},
                "seal7": {"colors": [(127, 0, 255), (0, 255, 127), (127, 255, 0)], "effect": "gradient", "speed": 0.5},
                "seal8": {"colors": [(255, 255, 0), (0, 0, 255)], "effect": "blink", "speed": 1},
                "seal9": {"colors": [(0, 255, 0), (255, 0, 0)], "effect": "gradient", "speed": 1},
                "seal10": {"colors": [(127, 0, 255), (255, 0, 127)], "effect": "blink", "speed": 0.5},
                "seal11": {"colors": [(0, 255, 255), (255, 255, 0)], "effect": "gradient", "speed": 0.5},
                "seal12": {"colors": [(255, 127, 0), (127, 255, 0)], "effect": "blink", "speed": 1},
                "seal13": {"colors": [(255, 255, 255), (127, 127, 127)], "effect": "gradient", "speed": 1},
                "seal14": {"colors": [(255, 0, 255), (0, 255, 0)], "effect": "blink", "speed": 0.5},
                "seal15": {"colors": [(255, 255, 0), (0, 255, 255)], "effect": "gradient", "speed": 0.5},
                "seal16": {"colors": [(127, 0, 255), (0, 127, 255)], "effect": "blink", "speed": 1},
                "seal17": {"colors": [(255, 0, 127), (127, 0, 255)], "effect": "gradient", "speed": 1},
                "seal18": {"colors": [(255, 255, 255), (0, 0, 0)], "effect": "slide"},
            }

            if seal in seal_effects:
                color_pattern = seal_effects[seal]["colors"]
                effect = seal_effects[seal].get("effect", "static")
                # speed = seal_effects[seal].get("speed", 0)

                if effect == "static":
                    for i in range(LED_COUNT):
                        strip.setPixelColor(i, Color(color_pattern[i % len(color_pattern)][0],
                                                    color_pattern[i % len(color_pattern)][1],
                                                    color_pattern[i % len(color_pattern)][2]))
                    strip.show()
                elif effect == "slide":
                    tmp_count = count % LED_COUNT
                    for i in range(LED_COUNT):
                        if i == tmp_count:
                            strip.setPixelColor(i, Color(color_pattern[0][0],
                                                        color_pattern[0][1],
                                                        color_pattern[0][2]))
                        else:
                            strip.setPixelColor(i, Color(color_pattern[1][0],
                                                        color_pattern[1][1],
                                                        color_pattern[1][2]))

                    strip.show()
        
                elif effect == "blink":
                    tmp_count = count % 4
                    if tmp_count == 0 or tmp_count == 1:
                        for i in range(LED_COUNT):
                            strip.setPixelColor(i, Color(color_pattern[i % len(color_pattern)][0],
                                                        color_pattern[i % len(color_pattern)][1],
                                                        color_pattern[i % len(color_pattern)][2]))
                        strip.show()
                    if tmp_count == 2 or tmp_count == 3:
                        for i in range(LED_COUNT):
                            strip.setPixelColor(i, Color(0, 0, 0))  # Set all colors to black (off)
                        strip.show()
                elif effect == "gradient":
                    for i in range(LED_COUNT):
                        index = (i + round(time.time() * 3)) % len(color_pattern)
                        strip.setPixelColor(i, Color(color_pattern[index][0],
                                                    color_pattern[index][1],
                                                    color_pattern[index][2]))
                    strip.show()
            else:
                print(f"Unknown seal: {seal}")

        time.sleep(100/1000)

def clear_color(strip):
    """Clear LED color"""
    for i in range(strip.numPixels()):
        strip.setPixelColor(i, Color(0, 0, 0))
        strip.show()


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)
    parser.add_argument("--file", type=str, default=None)

    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--skip_frame", type=int, default=0)

    parser.add_argument(
        "--model",
        type=str,
        default='model/yolox/yolox_nano.onnx',
    )
    parser.add_argument(
        '--input_shape',
        type=str,
        default="416,416",
        help="Specify an input shape for inference.",
    )
    parser.add_argument(
        '--score_th',
        type=float,
        default=0.7,
        help='Class confidence',
    )
    parser.add_argument(
        '--nms_th',
        type=float,
        default=0.45,
        help='NMS IoU threshold',
    )
    parser.add_argument(
        '--nms_score_th',
        type=float,
        default=0.1,
        help='NMS Score threshold',
    )
    parser.add_argument(
        "--with_p6",
        action="store_true",
        help="Whether your model uses p6 in FPN/PAN.",
    )

    parser.add_argument("--sign_interval", type=float, default=2.0)
    parser.add_argument("--jutsu_display_time", type=int, default=5)

    parser.add_argument("--use_display_score", type=bool, default=False)
    parser.add_argument("--erase_bbox", type=bool, default=False)
    parser.add_argument("--use_jutsu_lang_en", type=bool, default=False)

    parser.add_argument("--chattering_check", type=int, default=1)

    parser.add_argument("--use_fullscreen", type=bool, default=False)

    args = parser.parse_args()

    return args


def main(seal_type):
    # 引数解析 #################################################################
    args = get_args()

    cap_width = args.width
    cap_height = args.height
    cap_device = args.device
    if args.file is not None:  # 動画ファイルを利用する場合
        cap_device = args.file

    fps = args.fps
    skip_frame = args.skip_frame

    model_path = args.model
    input_shape = tuple(map(int, args.input_shape.split(',')))
    score_th = args.score_th
    nms_th = args.nms_th
    nms_score_th = args.nms_score_th
    with_p6 = args.with_p6

    sign_interval = args.sign_interval
    jutsu_display_time = args.jutsu_display_time

    use_display_score = args.use_display_score
    erase_bbox = args.erase_bbox
    use_jutsu_lang_en = args.use_jutsu_lang_en

    chattering_check = args.chattering_check

    use_fullscreen = args.use_fullscreen

    # カメラ準備 ###############################################################
    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    # モデル読み込み ############################################################
    yolox = YoloxONNX(
        model_path=model_path,
        input_shape=input_shape,
        class_score_th=score_th,
        nms_th=nms_th,
        nms_score_th=nms_score_th,
        with_p6=with_p6,
        # providers=['CPUExecutionProvider'],
    )

    # FPS計測モジュール #########################################################
    cvFpsCalc = CvFpsCalc()

    # フォント読み込み ##########################################################
    # https://opentype.jp/kouzanmouhitufont.htm
    font_path = './utils/font/衡山毛筆フォント.ttf'

    # ラベル読み込み ###########################################################
    with open('setting/labels.csv', encoding='utf8') as f:  # 印
        labels = csv.reader(f)
        labels = [row for row in labels]

    with open('setting/jutsu.csv', encoding='utf8') as f:  # 術
        jutsu = csv.reader(f)
        jutsu = [row for row in jutsu]

    # 印の表示履歴および、検出履歴 ##############################################
    sign_max_display = 18
    sign_max_history = 44
    sign_display_queue = deque(maxlen=sign_max_display)
    sign_history_queue = deque(maxlen=sign_max_history)

    chattering_check_queue = deque(maxlen=chattering_check)
    for index in range(-1, -1 - chattering_check, -1):
        chattering_check_queue.append(index)

    # 術名の言語設定 ###########################################################
    lang_offset = 0
    jutsu_font_size_ratio = sign_max_display
    if use_jutsu_lang_en:
        lang_offset = 1
        jutsu_font_size_ratio = int((sign_max_display / 3) * 4)

    # その他変数初期化 #########################################################
    sign_interval_start = 0  # 印のインターバル開始時間初期化
    jutsu_index = 0  # 術表示名のインデックス
    jutsu_start_time = 0  # 術名表示の開始時間初期化
    frame_count = 0  # フレームナンバーカウンタ

    window_name = 'NARUTO HandSignDetection Ninjutsu Demo'
    if use_fullscreen:
        cv.namedWindow(window_name, cv.WINDOW_NORMAL)

    while True:
        start_time = time.time()

        # カメラキャプチャ #####################################################
        ret, frame = cap.read()
        if not ret:
            continue
        frame_count += 1
        debug_image = copy.deepcopy(frame)

        if (frame_count % (skip_frame + 1)) != 0:
            continue

        # FPS計測 ##############################################################
        fps_result = cvFpsCalc.get()

        # 検出実施 #############################################################
        bboxes, scores, class_ids = yolox.inference(frame)

        # 検出内容の履歴追加 ####################################################
        for _, score, class_id in zip(bboxes, scores, class_ids):
            class_id = int(class_id) + 1

            # 検出閾値未満の結果は捨てる
            if score < score_th:
                continue

            # 指定回数以上、同じ印が続いた場合に、印検出とみなす ※瞬間的な誤検出対策
            chattering_check_queue.append(class_id)
            if len(set(chattering_check_queue)) != 1:
                continue

            # 前回と異なる印の場合のみキューに登録
            if len(sign_display_queue
                   ) == 0 or sign_display_queue[-1] != class_id:
                sign_display_queue.append(class_id)
                sign_history_queue.append(class_id)
                sign_interval_start = time.time()  # 印の最終検出時間

        # 前回の印検出から指定時間が経過した場合、履歴を消去 ####################
        if (time.time() - sign_interval_start) > sign_interval:
            sign_display_queue.clear()
            sign_history_queue.clear()

        # 術成立判定 #########################################################
        jutsu_index, jutsu_start_time = check_jutsu(
            sign_history_queue,
            labels,
            jutsu,
            jutsu_index,
            jutsu_start_time,
        )

        # キー処理 ###########################################################
        key = cv.waitKey(1)
        if key == 99:  # C：印の履歴を消去
            sign_display_queue.clear()
            sign_history_queue.clear()
        if key == 27:  # ESC：プログラム終了
            seal_type.value = -1 # clear led
            break

        # 画面反映 #############################################################
        debug_image, seal_result = draw_debug_image(
            debug_image,
            font_path,
            fps_result,
            labels,
            bboxes,
            scores,
            class_ids,
            score_th,
            erase_bbox,
            use_display_score,
            jutsu,
            sign_display_queue,
            sign_max_display,
            jutsu_display_time,
            jutsu_font_size_ratio,
            lang_offset,
            jutsu_index,
            jutsu_start_time,
        )
        if use_fullscreen:
            cv.setWindowProperty(window_name, cv.WND_PROP_FULLSCREEN,
                                 cv.WINDOW_FULLSCREEN)
        cv.imshow(window_name, debug_image)
        # cv.moveWindow(window_name, 100, 100)

        seal_type.value = seal_result       

        # FPS調整 #############################################################
        elapsed_time = time.time() - start_time
        sleep_time = max(0, ((1.0 / fps) - elapsed_time))
        time.sleep(sleep_time)

    cap.release()
    cv.destroyAllWindows()


def check_jutsu(
    sign_history_queue,
    labels,
    jutsu,
    jutsu_index,
    jutsu_start_time,
):
    # 印の履歴から術名をマッチング
    sign_history = ''
    if len(sign_history_queue) > 0:
        for sign_id in sign_history_queue:
            sign_history = sign_history + labels[sign_id][1]
        for index, signs in enumerate(jutsu):
            if sign_history == ''.join(signs[4:]):
                jutsu_index = index
                jutsu_start_time = time.time()  # 術の最終検出時間
                break

    return jutsu_index, jutsu_start_time


def draw_debug_image(
    debug_image,
    font_path,
    fps_result,
    labels,
    bboxes,
    scores,
    class_ids,
    score_th,
    erase_bbox,
    use_display_score,
    jutsu,
    sign_display_queue,
    sign_max_display,
    jutsu_display_time,
    jutsu_font_size_ratio,
    lang_offset,
    jutsu_index,
    jutsu_start_time,
):
    out_label_numb = 0

    frame_width, frame_height = debug_image.shape[1], debug_image.shape[0]

    # 印のバウンディングボックスの重畳表示(表示オプション有効時) ###################
    if not erase_bbox:
        for bbox, score, class_id in zip(bboxes, scores, class_ids):
            class_id = int(class_id) + 1

            # 検出閾値未満のバウンディングボックスは捨てる
            if score < score_th:
                continue

            x1, y1 = int(bbox[0]), int(bbox[1])
            x2, y2 = int(bbox[2]), int(bbox[3])

            # バウンディングボックス(長い辺にあわせて正方形を表示)
            x_len = x2 - x1
            y_len = y2 - y1
            square_len = x_len if x_len >= y_len else y_len
            square_x1 = int(((x1 + x2) / 2) - (square_len / 2))
            square_y1 = int(((y1 + y2) / 2) - (square_len / 2))
            square_x2 = square_x1 + square_len
            square_y2 = square_y1 + square_len
            cv.rectangle(debug_image, (square_x1, square_y1),
                         (square_x2, square_y2), (255, 255, 255), 4)
            cv.rectangle(debug_image, (square_x1, square_y1),
                         (square_x2, square_y2), (0, 0, 0), 2)

            # 印の種類
            font_size = int(square_len / 2)
            debug_image = CvDrawText.puttext(
                debug_image, labels[class_id][1],
                (square_x2 - font_size, square_y2 - font_size), font_path,
                font_size, (185, 0, 0))

            # 検出スコア(表示オプション有効時)
            if use_display_score:
                font_size = int(square_len / 8)
                debug_image = CvDrawText.puttext(
                    debug_image, '{:.3f}'.format(score),
                    (square_x1 + int(font_size / 4),
                     square_y1 + int(font_size / 4)), font_path, font_size,
                    (185, 0, 0))

            # ラベル
            out_label_numb = class_id

    # ヘッダー作成：FPS #########################################################
    header_image = np.zeros((int(frame_height / 18), frame_width, 3), np.uint8)
    header_image = CvDrawText.puttext(header_image, "FPS:" + str(fps_result),
                                      (5, 0), font_path,
                                      int(frame_height / 20), (255, 255, 255))

    # フッター作成：印の履歴、および、術名表示 ####################################
    footer_image = np.zeros((int(frame_height / 10), frame_width, 3), np.uint8)

    # 印の履歴文字列生成
    sign_display = ''
    if len(sign_display_queue) > 0:
        for sign_id in sign_display_queue:
            sign_display = sign_display + labels[sign_id][1]

    # 術名表示(指定時間描画)
    if lang_offset == 0:
        separate_string = '・'
    else:
        separate_string = '：'
    if (time.time() - jutsu_start_time) < jutsu_display_time:
        if jutsu[jutsu_index][0] == '':  # 属性(火遁等)の定義が無い場合
            jutsu_string = jutsu[jutsu_index][2 + lang_offset]
        else:  # 属性(火遁等)の定義が有る場合
            jutsu_string = jutsu[jutsu_index][0 + lang_offset] + \
                separate_string + jutsu[jutsu_index][2 + lang_offset]
        footer_image = CvDrawText.puttext(
            footer_image, jutsu_string, (5, 0), font_path,
            int(frame_width / jutsu_font_size_ratio), (255, 255, 255))
    # 印表示
    else:
        footer_image = CvDrawText.puttext(footer_image, sign_display, (5, 0),
                                          font_path,
                                          int(frame_width / sign_max_display),
                                          (255, 255, 255))

    # ヘッダーとフッターをデバッグ画像へ結合 ######################################
    debug_image = cv.vconcat([header_image, debug_image])
    debug_image = cv.vconcat([debug_image, footer_image])

    return debug_image, out_label_numb


if __name__ == '__main__':
    seal_type = Value('i', 0)

    process1 = Process(target=main, args=[seal_type])
    process2 = Process(target=display_color_pattern, args=[seal_type])

    process1.start()
    process2.start()
