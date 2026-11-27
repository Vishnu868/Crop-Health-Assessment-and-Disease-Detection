// ESP32-CAM (AI-Thinker): capture a JPEG every CAPTURE_INTERVAL_MS (or on button press),
// base64-encode it and POST {"image": "<base64>"} to the Flask /predict endpoint.
// The JSON reply (disease, confidence, advisory) is printed to the Serial Monitor.
// Board: "AI Thinker ESP32-CAM" (esp32 core >= 2.0). Libraries: ArduinoJson >= 6.
// NOTE: provided as source; not compiled or flashed in the rebuild environment.
#include <WiFi.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>
#include "esp_camera.h"
#include "base64.h"

const char* WIFI_SSID = "YOUR_WIFI";
const char* WIFI_PASS = "YOUR_PASSWORD";
const char* SERVER_URL = "http://192.168.1.10:5000/predict";   // Flask server on the LAN
const uint32_t CAPTURE_INTERVAL_MS = 60UL * 1000UL;
const int BUTTON_PIN = 13;   // optional push button to GND

// AI-Thinker pin map
#define PWDN_GPIO_NUM 32
#define RESET_GPIO_NUM -1
#define XCLK_GPIO_NUM 0
#define SIOD_GPIO_NUM 26
#define SIOC_GPIO_NUM 27
#define Y9_GPIO_NUM 35
#define Y8_GPIO_NUM 34
#define Y7_GPIO_NUM 39
#define Y6_GPIO_NUM 36
#define Y5_GPIO_NUM 21
#define Y4_GPIO_NUM 19
#define Y3_GPIO_NUM 18
#define Y2_GPIO_NUM 5
#define VSYNC_GPIO_NUM 25
#define HREF_GPIO_NUM 23
#define PCLK_GPIO_NUM 22
#define FLASH_LED 4

bool initCamera() {
  camera_config_t c = {};
  c.ledc_channel = LEDC_CHANNEL_0; c.ledc_timer = LEDC_TIMER_0;
  c.pin_d0 = Y2_GPIO_NUM; c.pin_d1 = Y3_GPIO_NUM; c.pin_d2 = Y4_GPIO_NUM; c.pin_d3 = Y5_GPIO_NUM;
  c.pin_d4 = Y6_GPIO_NUM; c.pin_d5 = Y7_GPIO_NUM; c.pin_d6 = Y8_GPIO_NUM; c.pin_d7 = Y9_GPIO_NUM;
  c.pin_xclk = XCLK_GPIO_NUM; c.pin_pclk = PCLK_GPIO_NUM; c.pin_vsync = VSYNC_GPIO_NUM; c.pin_href = HREF_GPIO_NUM;
  c.pin_sccb_sda = SIOD_GPIO_NUM; c.pin_sccb_scl = SIOC_GPIO_NUM; c.pin_pwdn = PWDN_GPIO_NUM; c.pin_reset = RESET_GPIO_NUM;
  c.xclk_freq_hz = 20000000; c.pixel_format = PIXFORMAT_JPEG;
  c.frame_size = FRAMESIZE_VGA;     // 640x480 is plenty: the model works at 128-256 px
  c.jpeg_quality = 12; c.fb_count = 1;
  return esp_camera_init(&c) == ESP_OK;
}

void connectWiFi() {
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  Serial.print("WiFi");
  for (int i = 0; i < 40 && WiFi.status() != WL_CONNECTED; i++) { delay(500); Serial.print("."); }
  Serial.println(WiFi.status() == WL_CONNECTED ? " connected: " + WiFi.localIP().toString() : " FAILED");
}

void captureAndSend() {
  digitalWrite(FLASH_LED, HIGH); delay(150);
  camera_fb_t* fb = esp_camera_fb_get();
  digitalWrite(FLASH_LED, LOW);
  if (!fb) { Serial.println("capture failed"); return; }
  String b64 = base64::encode(fb->buf, fb->len);
  esp_camera_fb_return(fb);

  HTTPClient http;
  http.begin(SERVER_URL);
  http.addHeader("Content-Type", "application/json");
  http.setTimeout(20000);
  int code = http.POST("{\"image\":\"" + b64 + "\"}");
  if (code == 200) {
    DynamicJsonDocument doc(4096);
    if (!deserializeJson(doc, http.getString())) {
      Serial.printf("Disease: %s (%.1f%%)\n", doc["disease"].as<const char*>(), 100.0 * doc["confidence"].as<float>());
      Serial.printf("Fertilizer: %s\nPesticide: %s\nPrecautions: %s\n", doc["fertilizer"].as<const char*>(),
                    doc["pesticide"].as<const char*>(), doc["precautions"].as<const char*>());
    }
  } else {
    Serial.printf("HTTP error %d\n", code);
  }
  http.end();
}

void setup() {
  Serial.begin(115200);
  pinMode(FLASH_LED, OUTPUT);
  pinMode(BUTTON_PIN, INPUT_PULLUP);
  if (!initCamera()) { Serial.println("camera init failed"); ESP.restart(); }
  connectWiFi();
}

uint32_t lastShot = 0;
void loop() {
  if (WiFi.status() != WL_CONNECTED) connectWiFi();
  if (digitalRead(BUTTON_PIN) == LOW || millis() - lastShot > CAPTURE_INTERVAL_MS) {
    lastShot = millis();
    captureAndSend();
  }
  delay(50);
}
