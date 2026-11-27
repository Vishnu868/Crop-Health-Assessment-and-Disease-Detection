// NodeMCU (ESP8266): soil moisture + DHT11 -> local hysteresis pump control via relay,
// and reports readings to the Flask server (POST /api/sensors). Same logic as src/irrigation.py.
// Libraries: DHT sensor library (Adafruit), ArduinoJson >= 6.
// NOTE: provided as source; not compiled or flashed in the rebuild environment.
#include <ESP8266WiFi.h>
#include <ESP8266HTTPClient.h>
#include <ArduinoJson.h>
#include <DHT.h>

const char* WIFI_SSID = "YOUR_WIFI";
const char* WIFI_PASS = "YOUR_PASSWORD";
const char* SERVER_URL = "http://192.168.1.10:5000/api/sensors";

#define DHT_PIN D4
#define RELAY_PIN D1          // relay IN (active LOW on most modules)
#define MOISTURE_PIN A0
DHT dht(DHT_PIN, DHT11);

const float ON_THRESHOLD = 35.0, OFF_THRESHOLD = 60.0;  // % soil moisture
const unsigned long MAX_ON_MS = 600000UL;                // safety cut-off: 10 min
const int ADC_DRY = 1023, ADC_WET = 420;                  // calibrate for your sensor (ESP8266 ADC is 10-bit)
bool pumpOn = false; unsigned long onSince = 0;

float moisturePct() {
  int raw = analogRead(MOISTURE_PIN);
  float pct = (float)(ADC_DRY - raw) / (ADC_DRY - ADC_WET) * 100.0;
  return constrain(pct, 0, 100);
}

void setPump(bool on) {
  pumpOn = on;
  if (on) onSince = millis();
  digitalWrite(RELAY_PIN, on ? LOW : HIGH);
}

void setup() {
  Serial.begin(115200);
  pinMode(RELAY_PIN, OUTPUT);
  setPump(false);
  dht.begin();
  WiFi.begin(WIFI_SSID, WIFI_PASS);
}

void loop() {
  float m = moisturePct(), t = dht.readTemperature(), h = dht.readHumidity();
  if (isnan(t) || isnan(h)) { t = 25; h = 50; }  // fall back if the DHT read fails
  float onTh = ON_THRESHOLD + ((t > 32 && h < 40) ? 5.0 : 0.0);
  if (!pumpOn && m < onTh) setPump(true);
  else if (pumpOn && (m > OFF_THRESHOLD || millis() - onSince > MAX_ON_MS)) setPump(false);
  Serial.printf("moisture=%.0f%% temp=%.1fC hum=%.0f%% pump=%s\n", m, t, h, pumpOn ? "ON" : "OFF");

  if (WiFi.status() == WL_CONNECTED) {
    WiFiClient client; HTTPClient http;
    http.begin(client, SERVER_URL);
    http.addHeader("Content-Type", "application/json");
    char body[128];
    snprintf(body, sizeof(body), "{\"moisture\":%.1f,\"temperature\":%.1f,\"humidity\":%.1f}", m, t, h);
    http.POST(body);
    http.end();
  }
  delay(5000);
}
