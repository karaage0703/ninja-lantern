/*
 * main.ino
 * Copyright (C) 2023 MATSUOKA Takashi <matsujirushi@live.jp>
 * MIT License
 */

// {"command":"clear"}
// {"command":"setPixelColor","index":0,"red":5,"green":0,"blue":0}
// {"command":"setPixelColor","index":0,"red":0,"green":5,"blue":0}
// {"command":"setPixelColor","index":0,"red":0,"green":0,"blue":5}
// {"command":"show"}

////////////////////////////////////////////////////////////////////////////////
// Includes

#include <Adafruit_NeoPixel.h>
#include <ArduinoJson.h>

////////////////////////////////////////////////////////////////////////////////
// Constants

static constexpr uint16_t PIXEL_PIN = D10;
static constexpr uint16_t PIXEL_NUM = 10;

////////////////////////////////////////////////////////////////////////////////
// Variables

static Adafruit_NeoPixel Pixels{PIXEL_NUM, PIXEL_PIN, NEO_GRB + NEO_KHZ800};
static StaticJsonDocument<200> JsonDoc;

////////////////////////////////////////////////////////////////////////////////
// Helper functions for Serial

static String SerialReadString;

static String SerialReadStringUntil2(const char terminator)
{
    while (true)
    {
        const auto c = Serial.read();
        if (c < 0)
        {
            return {};
        }
        else if (c == terminator)
        {
            const String str{SerialReadString};
            SerialReadString.clear();
            return str;
        }
        else
        {
            SerialReadString += static_cast<char>(c);
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
// setup and loop

void setup()
{
    Serial.setTimeout(500);
    Serial.begin(115200);

    Pixels.begin();
}

void loop()
{
    const auto commandStr = SerialReadStringUntil2('\n');
    if (!commandStr.isEmpty())
    {
        const auto jsonError = deserializeJson(JsonDoc, commandStr);
        if (!jsonError)
        {
            const String command = JsonDoc["command"];
            if (command.equals("clear"))
            {
                Pixels.clear();
                Serial.println("{\"result\":\"ok\"}");
            }
            else if (command.equals("setPixelColor"))
            {
                const int index = JsonDoc["index"];
                const int red = JsonDoc["red"];
                const int green = JsonDoc["green"];
                const int blue = JsonDoc["blue"];
                Pixels.setPixelColor(index, Pixels.Color(red, green, blue));
                Serial.println("{\"result\":\"ok\"}");
            }
            else if (command.equals("show"))
            {
                Pixels.show();
                Serial.println("{\"result\":\"ok\"}");
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
