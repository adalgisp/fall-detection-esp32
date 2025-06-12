#include <Arduino.h>
#include <WiFi.h>
#include <Wire.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <Firebase_ESP_Client.h>
#include "addons/TokenHelper.h"
#include "addons/RTDBHelper.h"
#include "model_data.h"
#include "norm_params.h"

#include <TensorFlowLite_ESP32.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "esp_task_wdt.h"

/* ---------- CẤU HÌNH ---------- */
#define DEVICE_ID         "pr1GoGBZ3EH2KjIonoDm"
#define ALERT_NUMBER      "+84825745257"
#define FIREBASE_API_KEY  "AIzaSyBwoVtmXhoe0X1H43yDXRBJ-4oGT-qrtdI"
#define DATABASE_URL      "https://emergency-4aecc-default-rtdb.asia-southeast1.firebasedatabase.app"
#define USER_EMAIL        "tramnguyen.03112003@gmail.com"
#define USER_PASSWORD     "031103"

const char* WIFI_SSIDS[]  = { "vu nam 202 l1", "A" };
const char* WIFI_PASSES[] = { "vunam20552l1", "23082003" };
const int   WIFI_COUNT    = sizeof(WIFI_SSIDS)/sizeof(WIFI_SSIDS[0]);

#define MODEM_RX_PIN   16
#define MODEM_TX_PIN   17
#define MODEM_BAUD     115200

#define BUTTON_PIN       12
#define STATUS_LED_PIN   5
#define BUZZER_PIN       18
#define FALL_LED_PIN     LED_BUILTIN

#define I2C_SDA 21
#define I2C_SCL 22
#define BEEP_DURATION_MS 3000  // 3 giây đầu dùng để beep

const uint32_t WIFI_RETRY_INTERVAL   = 5UL * 60UL * 1000UL;
const uint32_t UPDATE_INTERVAL_MS    = 5UL * 60UL * 1000UL;
const uint32_t DEBOUNCE_MS           = 50;
const uint32_t MEDIUM_MS             = 10000;

/* tham số ML */
constexpr int    kWindow      = 48;
constexpr int    kDim         = 6;
constexpr int    kStrideMs    = 24;
constexpr size_t kArenaSize   = 24 * 1024;
constexpr float  kThreshold   = 0.37f;
constexpr uint32_t kCooldownMs = 10000;

/* ---------- BIẾN TOÀN CỤC ---------- */
enum AlertState { ALERT_IDLE, ALERT_ACTIVE };
AlertState alertState = ALERT_IDLE;
uint32_t alertStart = 0;
uint32_t pauseUntil = 0;  // Dừng đọc/inference tới thời điểm này

struct { float lat, lon; } lastLoc{0,0};
bool wifiOK = false;
bool gpsOK  = false;

/* ---------- PROTOTYPES ---------- */
void    sendAT(const char* cmd);
String  readModem(uint32_t timeout = 800);
bool    sendSMS(const String& msg);
void    initGSM();
bool    getGPS(float &lat, float &lon);
void    readMPU(float *ac, float *gy);
inline void normalize(float *v);
void    pushDevice();
String  getLocation();
void    onTokenStatus(TokenInfo info);
void    sendAlertSMS(const String &msg);
bool    updateAlertWithLocation(const String &msg);
void    initTime();
void    initFirebase();
void    onTokenStatus(token_info_t info);  
String  getLocation();                  
/* phần cứng & Firebase */
HardwareSerial   modemSerial(2);
Adafruit_MPU6050 mpu;
FirebaseData     fbdo;
FirebaseAuth     auth;
FirebaseConfig   fbCfg;

/* TF-Lite */
static tflite::MicroErrorReporter error_reporter;
static uint8_t tensor_arena[kArenaSize];
static tflite::MicroInterpreter* interpreter;
static TfLiteTensor* input;
static TfLiteTensor* output;
static bool quantModel;
float win[kWindow][kDim];
int idx = 0;
bool full = false;

/* Button */
bool buttonState = LOW, lastButton = LOW;
uint32_t pressStart = 0, lastBeep = 0, lastDebounce = 0;

/* ---------- TIỆN ÍCH ---------- */
// ---- Định nghĩa initTime() ----
void initTime() {
  esp_task_wdt_reset();
  // Cấu hình NTP (UTC+7)
  configTime(7 * 3600, 0, "pool.ntp.org", "time.google.com");
}

// ---- Định nghĩa initFirebase() ----
void initFirebase() {
  esp_task_wdt_reset();
  fbCfg.api_key               = FIREBASE_API_KEY;
  fbCfg.database_url          = DATABASE_URL;
  auth.user.email             = USER_EMAIL;
  auth.user.password          = USER_PASSWORD;
  fbCfg.token_status_callback = onTokenStatus;
  Firebase.begin(&fbCfg, &auth);
}
void onTokenStatus(token_info_t info) {
  Serial.printf("Firebase token status: %d\n", info);
}

void sendAT(const char* cmd) {
  esp_task_wdt_reset();
  while (modemSerial.available()) modemSerial.read();
  modemSerial.println(cmd);
}

String readModem(uint32_t timeout) {
  esp_task_wdt_reset();
  String s;
  uint32_t t0 = millis();
  while (millis() - t0 < timeout) {
    esp_task_wdt_reset();
    while (modemSerial.available()) {
      s += char(modemSerial.read());
    }
    yield();
  }
  return s;
}

String getLocation() {
  if (gpsOK) {
    char buf[32];
    snprintf(buf, sizeof(buf), "GPS:%.6f,%.6f", lastLoc.lat, lastLoc.lon);
    return String(buf);
  }
  return "Unknown";
}

// Kiểm tra đăng ký mạng trước khi gửi SMS
bool waitForRegistration(uint32_t timeout = 15000) {
  uint32_t t0 = millis();
  while (millis() - t0 < timeout) {
    sendAT("AT+CREG?");
    String r = readModem(500);
    Serial.printf("CREG→ %s\n", r.c_str());
    // Nếu stat = 1 (home) hoặc 5 (roaming), regardless of n:
    if (r.indexOf(",1") >= 0 || r.indexOf(",5") >= 0) {
      return true;
    }
    delay(500);
  }
  return false;
}


// Kiểm tra độ mạnh tín hiệu (0..31, 99=unknown)
int getSignalQuality() {
  sendAT("AT+CSQ");
  String r = readModem(500);
  Serial.printf("CSQ→ %s\n", r.c_str());
  int comma = r.indexOf(',');
  if (r.indexOf("+CSQ: ") >= 0 && comma >= 0) {
    int val = r.substring(r.indexOf(": ") + 2, comma).toInt();
    return val;
  }
  return -1;
}

// Gửi SMS với retry
bool sendSMS(const String& msg) {
  const int MAX_RETRY = 3;
  for (int attempt = 1; attempt <= MAX_RETRY; attempt++) {
    // 1) Chờ đăng ký mạng
    if (!waitForRegistration(5000)) {
      Serial.printf("sendSMS: Chưa đăng ký mạng, retry %d/%d\n", attempt, MAX_RETRY);
      delay(2000);
      continue;
    }
    // 2) Kiểm tra CSQ
    int csq = getSignalQuality();
    if (csq < 5 || csq > 31) {
      Serial.printf("sendSMS: Tín hiệu yếu (CSQ=%d), retry %d/%d\n", csq, attempt, MAX_RETRY);
      delay(2000);
      continue;
    }

    // 3) Text mode
    sendAT("AT+CMGF=1");
    {
      uint32_t t0 = millis();
      while (millis() - t0 < 2000) {
        String resp = readModem(200);
        if (resp.indexOf("OK") >= 0) break;
      }
    }

    // 4) Charset
    sendAT("AT+CSCS=\"GSM\"");
    {
      uint32_t t0 = millis();
      while (millis() - t0 < 2000) {
        String resp = readModem(200);
        if (resp.indexOf("OK") >= 0) break;
      }
    }

    // 5) Lệnh CMGS
    sendAT((String("AT+CMGS=\"") + ALERT_NUMBER + "\"").c_str());

    // 6) Chờ prompt '>'
    {
      bool gotPrompt = false;
      uint32_t t0 = millis();
      while (millis() - t0 < 5000) {
        String resp = readModem(200);
        if (resp.indexOf(">") >= 0) {
          gotPrompt = true;
          break;
        }
      }
      if (!gotPrompt) {
        Serial.println("sendSMS: Không nhận được '>' từ modem.");
        continue;
      }
    }

    // 7) Gửi nội dung + Ctrl+Z
    modemSerial.print(msg);
    modemSerial.write(0x1A);
    modemSerial.flush();
    Serial.println("sendSMS: Gửi xong, chờ reply…");

    // 8) Chờ kết quả (max 20s)
    {
      uint32_t t0 = millis();
      while (millis() - t0 < 20000) {
        if (modemSerial.available()) {
          String line = modemSerial.readStringUntil('\n');
          line.trim();
          if (line.indexOf("+CMGS:") >= 0 || line.indexOf("OK") >= 0) {
            Serial.println("sendSMS: Thành công");
            return true;
          }
          if (line.indexOf("ERROR") >= 0) {
            Serial.println("sendSMS: Thất bại");
            break;
          }
        }
        yield();
      }
    }
    // Thất bại, thử lại
    Serial.printf("sendSMS: Thất bại lần %d, thử lại\n", attempt);
    delay(2000);
  }
  Serial.println("sendSMS: Thử hết 3 lần vẫn thất bại.");
  return false;
}

void sendAlertSMS(const String &msg) {
  sendSMS(msg);
}

void initGSM() {
  esp_task_wdt_reset();
  modemSerial.begin(MODEM_BAUD, SERIAL_8N1, MODEM_RX_PIN, MODEM_TX_PIN);
  delay(500); esp_task_wdt_reset();
  sendAT("ATE0"); readModem();
  sendAT("AT+CMEE=2"); readModem();
}

bool getGPS(float &la, float &lo) {
  esp_task_wdt_reset();
  sendAT("AT+CGNSINF");
  String r = readModem();
  int p = r.indexOf("+CGNSINF:"); if (p < 0) return false;
  int c1 = r.indexOf(',', p);
  for (int i = 0; i < 3; i++) c1 = r.indexOf(',', c1 + 1);
  int c2 = r.indexOf(',', c1 + 1);
  la = r.substring(c1 + 1, c2).toFloat();
  c1 = c2; c2 = r.indexOf(',', c1 + 1);
  lo = r.substring(c1 + 1, c2).toFloat();
  return true;
}

void readMPU(float*ac,float*gy) {
  esp_task_wdt_reset();
  sensors_event_t a, g, t;
  mpu.getEvent(&a, &g, &t);
  const float LSB = 2048.0f, G_SI = 9.81f;
  ac[0] = a.acceleration.x * LSB / G_SI;
  ac[1] = a.acceleration.y * LSB / G_SI;
  ac[2] = a.acceleration.z * LSB / G_SI;
  gy[0] = g.gyro.x;
  gy[1] = g.gyro.y;
  gy[2] = g.gyro.z;
}

inline void normalize(float*v) {
  esp_task_wdt_reset();
  for (int i = 0; i < kDim; i++) {
    v[i] = (norm_std[i] < 1e-6f) ? 0.0f : (v[i] - norm_mean[i]) / norm_std[i];
  }
}
void updateUserStatus(bool needHelp, const String &reason) {
  // Lấy timestamp hiện tại (ISO 8601)
  struct tm tm;
  char ts[25];
  if (getLocalTime(&tm, 200)) {
    strftime(ts, sizeof(ts), "%Y-%m-%dT%H:%M:%SZ", &tm);
  } else {
    strcpy(ts, "");  // fallback
  }

  // Chuẩn bị JSON
  FirebaseJson statusJson;
  statusJson.add("needHelp", needHelp);
  statusJson.add("reason", reason);
  statusJson.add("reportedAt", String(ts));

  // Cập nhật nhánh userStatus
  String path = "/ESP32/Devices/" + String(DEVICE_ID) + "/userStatus";
  bool ok = Firebase.RTDB.setJSON(&fbdo, path.c_str(), &statusJson);
  if (!ok) {
    Serial.printf("updateUserStatus failed: %s\n", fbdo.errorReason().c_str());
    // Có thể thêm retry hoặc initFirebase() tùy ý
  } else {
    Serial.println("updateUserStatus: success");
  }
}

void pushDevice() {
  if (!wifiOK) return;

  // 1) Kiểm tra time đã hợp lệ chưa
  struct tm tm;
  if (!getLocalTime(&tm, 1000)) {
    Serial.println("pushDevice: Time chưa set, bỏ qua pushDevice()");
    return;
  }

  esp_task_wdt_reset();

  // 2) Chuẩn bị JSON
  char ts[20];
  strftime(ts, sizeof(ts), "%Y-%m-%d %H:%M:%S", &tm);
  FirebaseJson j;
  j.add("device_id", DEVICE_ID);
  j.add("lastSeen", ts);
  j.add("location", getLocation());
  j.add("status", "Active");

  // 3) Thực hiện push lên Firebase
  esp_task_wdt_delete(NULL);
  bool ok = Firebase.RTDB.setJSON(&fbdo, "/ESP32/Devices/" + String(DEVICE_ID), &j);
  esp_task_wdt_add(NULL);

  // 4) Xử lý kết quả
  if (ok) {
    Serial.println("pushDevice: Thành công");
    // Cập nhật userStatus mặc định mỗi lần pushDevice
     updateUserStatus(false, "is well");
  } else {
    String reason = fbdo.errorReason();
    Serial.printf("pushDevice: Thất bại: %s\n", reason.c_str());


    // Nếu lỗi liên quan SSL hoặc token expired, gọi lại initFirebase()
    if (reason.indexOf("ssl engine closed") >= 0 ||
        reason.indexOf("Failed to initlalize the SSL") >= 0 ||
        reason.indexOf("TOKEN_EXPIRED") >= 0 ||
        fbdo.httpCode() == 401) {
      Serial.println("pushDevice: Lỗi TLS/Token, gọi initFirebase()...");
      initFirebase();
      delay(500);
    }
  }
}

void initWiFi() {
  esp_task_wdt_reset();
  wifiOK = false;
  for (int i = 0; i < WIFI_COUNT; i++) {
    esp_task_wdt_reset();
    WiFi.begin(WIFI_SSIDS[i], WIFI_PASSES[i]);
    uint32_t st = millis();
    while (WiFi.status() != WL_CONNECTED && millis() - st < 10000) {
      esp_task_wdt_reset();
      delay(200); yield();
    }
    if (WiFi.status() == WL_CONNECTED) {
      esp_task_wdt_reset();
      wifiOK = true;
      initTime();
      initFirebase();
      break;
    }
  }
}

void setupModel() {
  esp_task_wdt_reset();
  const tflite::Model* m = tflite::GetModel(fall_model_tflite);
  static tflite::AllOpsResolver resolver;
  static tflite::MicroInterpreter interp(m, resolver, tensor_arena, kArenaSize, &error_reporter);
  interpreter = &interp;
  interpreter->AllocateTensors();
  input      = interpreter->input(0);
  output     = interpreter->output(0);
  quantModel = (input->type == kTfLiteInt8);
}

bool updateAlertWithLocation(const String &msg) {
  esp_task_wdt_reset();
  float la, lo;
  if (gpsOK || getGPS(la, lo)) {
    lastLoc = { la, lo };
    gpsOK   = true;
  }

  // Chỉ chờ NTP tối đa 200 ms
  struct tm tm;
  char ts[20] = "";
  if (getLocalTime(&tm, 200)) {
    strftime(ts, sizeof(ts), "%Y-%m-%d %H:%M:%S", &tm);
  } else {
    Serial.println("updateAlert: NTP chưa có, đẩy không có timestamp");
  }

  FirebaseJson j;
  if (strlen(ts) > 0) {
    j.set("createdAt", ts);
  }
  j.set("location",  getLocation());
  j.set("message",   msg);
  j.set("name",      "Test Device");
  j.set("read",      false);

  String path = String("/alerts/") + DEVICE_ID;

  esp_task_wdt_delete(NULL);
  bool ok = Firebase.RTDB.pushJSON(&fbdo, path.c_str(), &j);
  if (ok) {
    String newKey = fbdo.pushName();
    Serial.printf("Alert pushed, key = %s\n", newKey.c_str());
  } else {
    Serial.printf("Push alert failed: %s\n", fbdo.errorReason().c_str());
  }
  esp_task_wdt_add(NULL);

  return ok;
}
enum FallState { IDLE, PENDING_CONFIRM, CONFIRMED_FALL };
FallState fallState = IDLE;
uint32_t fallDetectedAt = 0;
bool cancelRequested = false;

// ==== Biến tạm điều khiển thời gian trong loop ====
static uint32_t lastStride       = 0;    // để giữ thời điểm chạy inference theo kStrideMs
static uint32_t lastBeepBlink    = 0;    // để điều khiển 3 lần beep (startBeepBlink3x)
static uint32_t lastEmergencyBlink = 0;  // để điều khiển nhấp nháy buzzer/LED 10s khi CONFIRMED_FALL
static bool     blinkState       = false; // trạng thái nhấp nháy buzzer/LED
static bool     blinking         = false; // flag đang thực hiện 3 lần beep
static uint32_t blinkCount       = 0;     // đếm số lần beep đã thực hiện

// ==== Hàm bắt đầu beep 3 lần (mỗi lần 300ms) ====
void startBeepBlink3x() {
  blinking = true;
  blinkCount = 0;
  lastBeepBlink = millis();
}

// ==== Hàm xử lý beep 3 lần không dùng delay() ====
void handleBeepBlink3x(uint32_t now) {
  if (!blinking) return;
  Serial.println("DEBUG: handleBeepBlink3x() đang chạy");
  if (now - lastBeepBlink >= 300) {
    // Toggle buzzer state
    bool buzzerState = digitalRead(BUZZER_PIN);
    digitalWrite(BUZZER_PIN, !buzzerState);
    lastBeepBlink = now;

    // Đếm mỗi khi chúng ta vừa tắt buzzer (tức là vừa hoàn thành 1 chu kỳ ON→OFF)
    if (!buzzerState) blinkCount++;

    // Nếu đã beep đủ 3 lần (3 chu kỳ ON→OFF), dừng
    if (blinkCount >= 3) {
      blinking = false;
      digitalWrite(BUZZER_PIN, LOW);
    }
  }
}

// ==== Hàm reset lại sliding window ====
void resetWindow() {
  idx = 0;
  full = false;
  memset(win, 0, sizeof(win));
}
// ----- safePushAlert: đẩy alert lên Firebase với retry nếu SSL đóng -----
#define FIREBASE_MAX_RETRY 2

bool safePushAlert(const String &msg) {
  // 1) Nếu Wi-Fi mất, cố gắng reconnect
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("safePushAlert: Wi-Fi mất, reconnect...");
    wifiOK = false;
    initWiFi();
    delay(500);
  }

  // 2) Đảm bảo có time chính xác (chờ thêm nếu cần)
  struct tm nowTm;
  if (!getLocalTime(&nowTm, 500)) {
    Serial.println("safePushAlert: Time chưa set, chờ thêm 1s...");
    delay(1000);
    if (!getLocalTime(&nowTm, 500)) {
      Serial.println("safePushAlert: Time vẫn chưa set, bỏ qua pushAlert.");
      return false;
    }
  }

  // 3) Gửi JSON lên Firebase, retry tối đa FIREBASE_MAX_RETRY lần
  for (int attempt = 1; attempt <= FIREBASE_MAX_RETRY; attempt++) {
    FirebaseJson j;
    char ts[20];
    strftime(ts, sizeof(ts), "%Y-%m-%d %H:%M:%S", &nowTm);
    j.set("createdAt", ts);
    j.set("location",  getLocation());
    j.set("message",   msg);
    j.set("name",      "Test Device");
    j.set("Read",      false);

    String path = String("/alerts/") + DEVICE_ID;

    esp_task_wdt_delete(NULL);
    bool ok = Firebase.RTDB.pushJSON(&fbdo, path.c_str(), &j);
    esp_task_wdt_add(NULL);

    if (ok) {
      Serial.printf("safePushAlert: Đẩy alert thành công (key=%s)\n", fbdo.pushName().c_str());
      return true;
    }

    // Nếu thất bại, in lý do
    String reason = fbdo.errorReason();
    Serial.printf("safePushAlert: Attempt %d thất bại: %s\n", attempt, reason.c_str());

    // Nếu lỗi SSL hoặc token expired, gọi lại initFirebase()
    if (reason.indexOf("ssl engine closed") >= 0 ||
        reason.indexOf("Failed to initlalize the SSL") >= 0 ||
        reason.indexOf("TOKEN_EXPIRED") >= 0 ||
        fbdo.httpCode() == 401) {
      Serial.println("safePushAlert: Lỗi TLS/Token, gọi initFirebase()...");
      initFirebase();    // Thay vì Firebase.reconnect()
      delay(500);
    }

    delay(500);
  }

  Serial.println("safePushAlert: Đẩy alert thất bại sau nhiều lần thử.");
  return false;
}


void setup() {
  Serial.begin(115200);
  esp_task_wdt_init(300, false);
  esp_task_wdt_add(NULL);

  pinMode(BUTTON_PIN, INPUT_PULLDOWN);
  pinMode(STATUS_LED_PIN, OUTPUT);
  pinMode(BUZZER_PIN, OUTPUT);
  pinMode(FALL_LED_PIN, OUTPUT);

  Wire.begin(I2C_SDA, I2C_SCL);
  mpu.begin();
  setupModel();

  initGSM();
  initWiFi();

  Serial.print("Waiting for NTP sync (max 2s)");
  struct tm timeinfo;
  uint32_t ntpStart = millis();
  while (!getLocalTime(&timeinfo) && (millis() - ntpStart < 8000)) {
    esp_task_wdt_reset();
    Serial.print(".");
    delay(200);
  }
  if (getLocalTime(&timeinfo)) {
    Serial.println(" done");
  } else {
    Serial.println(" timeout");
  }

  pushDevice();
}

void loop() {
  uint32_t now = millis();
  esp_task_wdt_reset();

  // ==========================
  // 1. Nhấp nháy buzzer + LED khi CONFIRMED_FALL (10s)
  // ==========================
  if (fallState == CONFIRMED_FALL && now < pauseUntil) {
    if (now - lastEmergencyBlink >= 500) {
      lastEmergencyBlink = now;
      blinkState = !blinkState;
      digitalWrite(BUZZER_PIN, blinkState);
      digitalWrite(STATUS_LED_PIN, blinkState);
    }
  } else {
    // Nếu không trong chu kỳ 10s CONFIRMED_FALL, đảm bảo tắt luôn
    digitalWrite(BUZZER_PIN, LOW);
    digitalWrite(STATUS_LED_PIN, LOW);
  }

  // Nếu đang pause (đang nhấp nháy cảnh báo CONFIRMED_FALL), chỉ nhấp nháy FALL_LED_PIN rồi return
  if (now < pauseUntil) {
    static uint32_t lastFallLed = 0;
    if (now - lastFallLed >= 500) {
      lastFallLed = now;
      digitalWrite(FALL_LED_PIN, !digitalRead(FALL_LED_PIN));
    }
    return;
  } else {
    digitalWrite(FALL_LED_PIN, LOW);
  }

  // ==========================
  // 2. Xử lý beep ngắn khi giữ nút (SOS)
  // ==========================
  static bool shortBeepActive = false;
  static uint32_t shortBeepStart = 0;

  // Nếu shortBeepActive đã bật, tắt buzzer sau 50 ms
  if (shortBeepActive && now - shortBeepStart >= 50) {
    digitalWrite(BUZZER_PIN, LOW);
    shortBeepActive = false;
  }

  // ==========================
  // 3. Xử lý nút nhấn (debounce + phân biệt PENDING_CONFIRM vs SOS)
  // ==========================
  bool rd = digitalRead(BUTTON_PIN);
  if (rd != lastButton) {
    lastDebounce = now;
  }
  if (now - lastDebounce > DEBOUNCE_MS && rd != buttonState) {
    buttonState = rd;

    if (buttonState) {
      // Bắt đầu giữ nút
      pressStart = now;
      lastBeep = now;
      digitalWrite(STATUS_LED_PIN, HIGH);
    } else {
      // Thả nút
      digitalWrite(STATUS_LED_PIN, LOW);
      uint32_t pressDuration = now - pressStart;

      if (fallState == PENDING_CONFIRM && !cancelRequested) {
        // Trong 3s xác nhận : huỷ cảnh báo ngã
        cancelRequested = true;
        fallState = IDLE;
        if (!safePushAlert("Người dùng xác nhận ổn sau phát hiện té. Huỷ cảnh báo.")) {
          Serial.println("safePushAlert: Gửi hủy cảnh báo thất bại.");
        }        
        resetWindow();
      }
      else if (pressDuration < MEDIUM_MS) {
        // Bấm ngắn (SOS) nếu không phải PENDING_CONFIRM
        updateAlertWithLocation("Người dùng thiết bị 01 bấm nút cấp cứu khẩn cấp!");
       // sendAlertSMS("NGUOI DUNG THIET BI 01 BAM NUT CAP CUU KHAN CAP! VUI LONG KIEM TRA NGAY LAP TUC!");
        pauseUntil = now + 10000;
        resetWindow();
      }
    }
  }
  lastButton = rd;

  // Nếu đang giữ nút, beep mỗi 1s 1 lần, mỗi lần duy trì 50 ms
  if (buttonState && now - lastBeep >= 1000) {
    digitalWrite(BUZZER_PIN, HIGH);
    shortBeepActive = true;
    shortBeepStart = now;
    lastBeep = now;
  }

  // ==========================
  // 4. Sliding window & inference cứ đúng kStrideMs
  // ==========================
  while (now - lastStride >= kStrideMs) {
    lastStride += kStrideMs;

    if (fallState == IDLE) {
      // 4.1. Đọc sensor & normalize
      float a[3], g[3], v[kDim];
      readMPU(a, g);
      for (int i = 0; i < 3; i++)       v[i]     = a[i];
      for (int i = 0; i < 3; i++)       v[3 + i] = g[i];
      normalize(v);

      // 4.2. Đưa vào sliding window
      for (int c = 0; c < kDim; c++) {
        win[idx][c] = v[c];
      }
      idx = (idx + 1) % kWindow;
      full = (idx == 0);

      // 4.3. Nếu window đầy, thực hiện inference
      if (full) {
        if (quantModel) {
          float s  = input->params.scale;
          int   zp = input->params.zero_point;
          for (int t = 0; t < kWindow; t++) {
            for (int c = 0; c < kDim; c++) {
              int q = lroundf(win[t][c] / s) + zp;
              if (q < -128) q = -128;
              if (q > 127)  q = 127;
              input->data.int8[t * kDim + c] = (int8_t)q;
            }
          }
        } else {
          for (int t = 0; t < kWindow; t++) {
            for (int c = 0; c < kDim; c++) {
              input->data.f[t * kDim + c] = win[t][c];
            }
          }
        }

        if (interpreter->Invoke() == kTfLiteOk) {
          float prob = quantModel
            ? (output->data.int8[0] - output->params.zero_point) * output->params.scale
            : output->data.f[0];
          Serial.printf("prob=%.3f\n", prob);
          if (prob > kThreshold && now >= alertStart + kCooldownMs) {
            // Khi phát hiện ngã: chuyển sang PENDING_CONFIRM
            alertStart      = now;
            fallDetectedAt  = now;
            fallState       = PENDING_CONFIRM;
            cancelRequested = false;

            if (!safePushAlert("Thiết bị phát hiện ngã. Đang chờ người dùng xác nhận...")) {
              Serial.println("safePushAlert: Gửi pending thất bại.");
            }      
            Serial.println("DEBUG: Fall detected → gọi startBeepBlink3x()");      
            startBeepBlink3x();
          }
        }
      }
    }
  } // end while sliding window

  // ==========================
  // 5. Nếu PENDING_CONFIRM quá 3s mà chưa huỷ thì CONFIRMED_FALL
  // ==========================
  if (fallState == PENDING_CONFIRM && now - fallDetectedAt >= MEDIUM_MS && !cancelRequested) {
    fallState = CONFIRMED_FALL;
    if (!safePushAlert("NGƯỜI DÙNG THIẾT BỊ 1 ĐÃ BỊ NGÃ! VUI LÒNG GIÚP ĐỠ")) {
      Serial.println("safePushAlert: Gửi confirmed thất bại.");
    }
    updateUserStatus(true, "fall detected");    
    // sendAlertSMS("NGUOI DUNG THIET BI 01 DA BI NGA. VUI LONG GIUP DO!!!");
    pauseUntil = now + 10000;
    // Nhấp nháy buzzer+LED 10s sẽ được handle ở bước 1

    resetWindow();
  }

  // ==========================
  // 6. Heartbeat & Wi-Fi retry
  // ==========================
  static uint32_t updTimer  = 0;
  static uint32_t wifiRetry = 0;

  if (wifiOK && now - updTimer > UPDATE_INTERVAL_MS) {
    pushDevice();
    updTimer = now;
  }
  if (!wifiOK && now - wifiRetry > WIFI_RETRY_INTERVAL) {
    initWiFi();
    wifiRetry = now;
  }

  // ==========================
  // 7. Gọi handleBeepBlink3x() cuối cùng
  // ==========================
  handleBeepBlink3x(now);
  
}
