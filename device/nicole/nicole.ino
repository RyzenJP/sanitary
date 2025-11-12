#include <WiFi.h>
#include <HTTPClient.h>
#include <WiFiManager.h>
#include <Preferences.h>

// === TDS Sensor Hardware Configuration ===
const int TDS_PIN = 36;  // GPIO36 (ADC0)

// === Turbidity Sensor Pin Setup ===
const int turbidityPin = 34;      // ADC input for Turbidity
const int ledGreenTurbidity = 25; // Good water LED for turbidity
const int ledYellowTurbidity = 26; // Warning LED for turbidity
const int ledRedTurbidity = 27;   // Bad water LED for turbidity

// === TDS Sensor LED Setup ===
const int ledGreenTDS = 12;       // Good water LED for TDS
const int ledYellowTDS = 13;      // Warning LED for TDS  
const int ledRedTDS = 14;         // Bad water LED for TDS

// === ADC Calibration ===
const float adcMax = 4095.0;   // 12-bit ADC (ESP32)
const float vRef = 3.3;        // ESP32 ADC reference voltage
const float dividerFactor = 3.0; // From 10k:20k divider (scales 0–4.5V to 0–3V)

// Water detection thresholds
const int DRY_SENSOR_THRESHOLD = 100;  // ADC value below this indicates no water
const float MIN_VOLTAGE = 0.1;         // Voltage below this indicates no water
const int CONSECUTIVE_DRY_READINGS = 3; // Number of dry readings before declaring no water

// Calibration parameters
const float VREF = 3.3;        // ADC reference voltage
const int ADC_RESOLUTION = 4095; // 12-bit resolution
const float TDS_FACTOR = 0.5;   // TDS conversion factor

// Server configuration
char serverUrl[150] = "https://192.168.137.134/sanitary/sensor/device/tds/tds_receiver.php";

// Timing and stability parameters
const unsigned long READING_INTERVAL = 30000; // 30 seconds between readings
const unsigned long CONNECTION_TIMEOUT = 30000; // 30 seconds connection timeout
const unsigned long WIFI_TIMEOUT = 10000;     // 10 seconds WiFi timeout
const int MAX_RETRIES = 3;

// Global variables
WiFiManager wm;
Preferences preferences;
unsigned long lastReadingTime = 0;
int connectionRetries = 0;
int dryCount = 0; // Counter for consecutive dry readings

// Custom calibration parameters (can be adjusted)
float calibrationOffset = 0.0;
float calibrationFactor = 1.0;

// Improved calibration with multiple points
const int NUM_CALIBRATION_POINTS = 3;
float calibrationKnownValues[NUM_CALIBRATION_POINTS] = {0.0, 500.0, 1000.0}; // Known TDS values for calibration
float calibrationMeasuredValues[NUM_CALIBRATION_POINTS] = {0.0, 0.0, 0.0}; // Measured values
int currentCalibrationPoint = 0;

// Add a variable to track the actual VREF (measured)
float measuredVref = VREF;

// Device ID for database tracking
String deviceId = "SENSOR_001"; // Changed to match your dashboard

// === Turbidity formula (WHO Standard Nephelometric) ===
float voltageToNTU(float v) {
  // WHO-compliant nephelometric turbidity calculation
  // Based on WHO Guidelines for Drinking-water Quality
  // NTU (Nephelometric Turbidity Units) measures light scattering at 90 degrees
  
  float ntu;
  
  // WHO Standard Ranges:
  // 0-1 NTU: Excellent (WHO preferred for drinking water)
  // 1-5 NTU: Acceptable (WHO maximum for drinking water)
  // >5 NTU: Concerning (WHO concerning level)
  // >50 NTU: Unsafe (WHO unsafe for consumption)
  
  // Nephelometric principle: Light scattering intensity is proportional to particle concentration
  // Higher voltage indicates more light scattering = higher turbidity
  
  if (v <= 0.05) {
    // Ultra-clear water (distilled/RO filtered)
    ntu = 0.0;
  } else if (v <= 0.5) {
    // Excellent quality (0-1 NTU range)
    ntu = v * 2.0; // 0.05V = 0.1 NTU, 0.5V = 1.0 NTU
  } else if (v <= 1.0) {
    // Good quality (1-2 NTU range)
    ntu = 1.0 + (v - 0.5) * 2.0; // 0.5V = 1.0 NTU, 1.0V = 2.0 NTU
  } else if (v <= 2.0) {
    // Acceptable quality (2-5 NTU range)
    ntu = 2.0 + (v - 1.0) * 3.0; // 1.0V = 2.0 NTU, 2.0V = 5.0 NTU
  } else if (v <= 3.0) {
    // Concerning quality (5-10 NTU range)
    ntu = 5.0 + (v - 2.0) * 5.0; // 2.0V = 5.0 NTU, 3.0V = 10.0 NTU
  } else if (v <= 4.0) {
    // High turbidity (10-25 NTU range)
    ntu = 10.0 + (v - 3.0) * 15.0; // 3.0V = 10.0 NTU, 4.0V = 25.0 NTU
  } else {
    // Very high turbidity (25+ NTU)
    ntu = 25.0 + (v - 4.0) * 25.0; // 4.0V = 25.0 NTU, 5.0V = 50.0 NTU
  }
  
  // WHO safety limits
  if (ntu < 0) ntu = 0;
  if (ntu > 100) ntu = 100; // Cap at 100 NTU (extremely unsafe)
  
  return ntu;
}

void setup() {
  Serial.begin(115200);
  delay(1000); // Allow serial to stabilize
  
  Serial.println("\n=== Water Quality Sensor Initialization ===");
  
  // Initialize turbidity LED pins
  pinMode(ledGreenTurbidity, OUTPUT);
  pinMode(ledYellowTurbidity, OUTPUT);
  pinMode(ledRedTurbidity, OUTPUT);
  digitalWrite(ledGreenTurbidity, LOW);
  digitalWrite(ledYellowTurbidity, LOW);
  digitalWrite(ledRedTurbidity, LOW);
  
  // Initialize TDS LED pins
  pinMode(ledGreenTDS, OUTPUT);
  pinMode(ledYellowTDS, OUTPUT);
  pinMode(ledRedTDS, OUTPUT);
  digitalWrite(ledGreenTDS, LOW);
  digitalWrite(ledYellowTDS, LOW);
  digitalWrite(ledRedTDS, LOW);
  
  // Initialize preferences for storing calibration data
  preferences.begin("tds-sensor", false);
  
  // Load calibration data if available
  calibrationOffset = preferences.getFloat("offset", 0.0);
  calibrationFactor = preferences.getFloat("factor", 1.0);
  measuredVref = preferences.getFloat("vref", VREF);
  
  // Initialize ADC
  analogReadResolution(12);
  analogSetAttenuation(ADC_11db);
  
  // Measure actual VREF for better accuracy
  measureVref();
  
  // Stabilize ADC readings by taking some dummy readings
  for (int i = 0; i < 20; i++) {
    analogRead(TDS_PIN);
    analogRead(turbidityPin);
    delay(2);
  }
  
  // Test sensor connections
  testSensorConnections();
  
  // Connect to WiFi
  setupWiFi();
  
  Serial.println("Initialization complete");
  Serial.printf("Measured VREF: %.3fV\n", measuredVref);
  Serial.println("Device ID: " + deviceId);
  Serial.println("TDS LEDs: Green=GPIO12, Yellow=GPIO13, Red=GPIO14");
  Serial.println("Turbidity LEDs: Green=GPIO25, Yellow=GPIO26, Red=GPIO27");
  Serial.println("==============================");
}

void testSensorConnections() {
  Serial.println("Testing sensor connections...");
  
  // Test TDS sensor
  int tdsTestValue = analogRead(TDS_PIN);
  Serial.printf("TDS Sensor Test Reading: %d\n", tdsTestValue);
  
  // Test Turbidity sensor
  int turbidityTestValue = analogRead(turbidityPin);
  Serial.printf("Turbidity Sensor Test Reading: %d\n", turbidityTestValue);
  
  if (tdsTestValue == 0 && turbidityTestValue == 0) {
    Serial.println("WARNING: Both sensors are reading 0. Check connections and power.");
    Serial.println("1. Ensure sensors are properly connected to 3.3V or 5V");
    Serial.println("2. Check ground connections");
    Serial.println("3. Verify sensor signal wires are connected to correct pins");
  }
}

void measureVref() {
  // Use a more reliable method to measure VREF
  // For ESP32, we can use the internal reference or assume 3.3V
  // The previous method was unreliable because it depends on external voltage
  
  // For now, use the standard ESP32 reference voltage
  measuredVref = 3.3;
  
  // If you want to calibrate VREF, connect a known voltage (like 3.3V from VCC) to a spare ADC pin
  // and measure it to get the actual reference voltage
  
  Serial.printf("Using VREF: %.3fV (standard ESP32 reference)\n", measuredVref);
  
  // Save the measured VREF
  preferences.putFloat("vref", measuredVref);
}

void setupWiFi() {
  WiFi.mode(WIFI_STA);
  WiFi.setAutoReconnect(true);
  WiFi.persistent(true);
  
  // Additional WiFi configuration for stability
  WiFi.setSleep(false); // Disable sleep for better connectivity
  
  Serial.println("Connecting to WiFi...");
  
  // Use WiFiManager with timeout
  wm.setConfigPortalTimeout(120);
  wm.setConnectTimeout(30);
  wm.setConnectRetries(5);
  
  if (!wm.autoConnect("TDS_Sensor_AP")) {
    Serial.println("Failed to connect and hit timeout");
    // Don't restart immediately, try to continue with offline readings
    Serial.println("Continuing in offline mode");
  } else {
    Serial.println("WiFi connected successfully");
    Serial.print("IP Address: ");
    Serial.println(WiFi.localIP());
  }
}

bool isSensorInWater(int analogValue, float voltage) {
  // Check if sensor is likely in water
  if (analogValue < DRY_SENSOR_THRESHOLD || voltage < MIN_VOLTAGE) {
    dryCount++;
    Serial.printf("Dry reading detected (%d/%d)\n", dryCount, CONSECUTIVE_DRY_READINGS);
    return false;
  } else {
    dryCount = 0; // Reset counter if we get a wet reading
    return true;
  }
}

float readTDS(int &rawAnalog, float &voltage) {
  // Take multiple samples for stability with improved filtering
  const int numSamples = 50;
  int samples[numSamples];
  
  // Collect samples
  for (int i = 0; i < numSamples; i++) {
    samples[i] = analogRead(TDS_PIN);
    delay(2);
  }
  
  // Sort samples for median filtering
  for (int i = 0; i < numSamples - 1; i++) {
    for (int j = i + 1; j < numSamples; j++) {
      if (samples[j] < samples[i]) {
        int temp = samples[i];
        samples[i] = samples[j];
        samples[j] = temp;
      }
    }
  }
  
  // Use median value to reduce noise
  rawAnalog = samples[numSamples / 2];
  
  // Calculate voltage using measured reference voltage
  voltage = rawAnalog * (measuredVref / ADC_RESOLUTION);
  
  // Check if sensor is in water
  if (!isSensorInWater(rawAnalog, voltage)) {
    return -1.0; // Special value indicating no water
  }
  
  // Calculate TDS value using a more accurate formula
  // This formula is more linear for common TDS ranges
  float tdsValue = (660.0 * voltage);
  
  // Apply calibration
  tdsValue = (tdsValue * calibrationFactor) + calibrationOffset;
  
  // Ensure non-negative value
  if (tdsValue < 0.0) {
    tdsValue = 0.0;
  }
  
  return tdsValue;
}

float readTurbidity(int &rawADC, float &voutESP32, float &sensorVoltage) {
  // Take multiple samples for stability
  const int numSamples = 20;
  int samples[numSamples];
  
  // Collect samples
  for (int i = 0; i < numSamples; i++) {
    samples[i] = analogRead(turbidityPin);
    delay(2);
  }
  
  // Sort samples for median filtering
  for (int i = 0; i < numSamples - 1; i++) {
    for (int j = i + 1; j < numSamples; j++) {
      if (samples[j] < samples[i]) {
        int temp = samples[i];
        samples[i] = samples[j];
        samples[j] = temp;
      }
    }
  }
  
  // Use median value to reduce noise
  rawADC = samples[numSamples / 2];
  
  // Calculate voltages
  voutESP32 = (rawADC / adcMax) * vRef;    // Voltage at ESP32 pin
  sensorVoltage = voutESP32 * dividerFactor;  // Back-calc to sensor output voltage

  // Check if sensor is in water (similar to TDS sensor)
  if (rawADC < DRY_SENSOR_THRESHOLD || sensorVoltage < MIN_VOLTAGE) {
    return -1.0; // Special value indicating no water
  }

  // === Convert to NTU ===
  float turbidityNTU = voltageToNTU(sensorVoltage);

  return turbidityNTU;
}

bool ensureWiFiConnection() {
  if (WiFi.status() == WL_CONNECTED) {
    return true;
  }
  
  Serial.println("WiFi disconnected. Attempting to reconnect...");
  
  WiFi.disconnect();
  delay(1000);
  WiFi.reconnect();
  
  unsigned long startTime = millis();
  while (WiFi.status() != WL_CONNECTED && (millis() - startTime) < WIFI_TIMEOUT) {
    delay(500);
    Serial.print(".");
  }
  
  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("\nWiFi reconnected successfully");
    connectionRetries = 0;
    return true;
  } else {
    Serial.println("\nFailed to reconnect to WiFi");
    return false;
  }
}

bool sendDataToServer(float tdsValue, int tdsAnalogValue, float tdsVoltage, 
                     float turbidityNTU, int turbidityRawADC, float voutESP32, float sensorVoltage) {
  if (!ensureWiFiConnection()) {
    Serial.println("Cannot send data - no WiFi connection");
    return false;
  }
  
  HTTPClient http;
  http.setTimeout(15000);
  http.setReuse(false);
  
  // Begin connection with HTTPS URL
  if (!http.begin(serverUrl)) {
    Serial.println("HTTP begin failed");
    http.end();
    return false;
  }
  
  // Create POST data with all parameters - now including both TDS and turbidity
  String postData = "tds_value=" + String(tdsValue, 2) + 
                   "&analog_value=" + String(tdsAnalogValue) + 
                   "&voltage=" + String(tdsVoltage, 3) + 
                   "&turbidity=" + String(turbidityNTU, 1) +
                   "&turbidity_raw_adc=" + String(turbidityRawADC) +
                   "&turbidity_vout_esp32=" + String(voutESP32, 3) +
                   "&turbidity_sensor_voltage=" + String(sensorVoltage, 3) +
                   "&device_id=" + deviceId;
  
  // Add headers
  http.addHeader("Content-Type", "application/x-www-form-urlencoded");
  http.addHeader("Connection", "close");
  
  // Send POST request
  int httpCode = http.POST(postData);
  bool success = false;
  
  if (httpCode > 0) {
    Serial.printf("HTTP response code: %d\n", httpCode);
    
    if (httpCode == HTTP_CODE_OK) {
      String payload = http.getString();
      Serial.println("Server response: " + payload);
      success = true;
      connectionRetries = 0;
    } else if (httpCode >= 300 && httpCode < 400) {
      // Handle redirect
      String redirectUrl = http.getLocation();
      Serial.println("Redirecting to: " + redirectUrl);
      
      http.end();
      HTTPClient httpRedirect;
      httpRedirect.begin(redirectUrl);
      httpRedirect.addHeader("Content-Type", "application/x-www-form-urlencoded");
      
      int redirectCode = httpRedirect.POST(postData);
      if (redirectCode == HTTP_CODE_OK) {
        success = true;
        connectionRetries = 0;
      }
      httpRedirect.end();
    }
  } else {
    Serial.printf("HTTP error: %s\n", http.errorToString(httpCode).c_str());
    connectionRetries++;
  }
  
  http.end();
  return success;
}

void updateTDSLEDs(float tdsValue) {
  // Turn off all TDS LEDs first
  digitalWrite(ledGreenTDS, LOW);
  digitalWrite(ledYellowTDS, LOW);
  digitalWrite(ledRedTDS, LOW);
  
  // If no water detected, all LEDs remain off
  if (tdsValue < 0) {
    return;
  }
  
  // Control TDS LEDs based on TDS value ranges
  if (tdsValue <= 150) {
    digitalWrite(ledGreenTDS, HIGH);   // Green ON - Excellent (0-150 ppm)
  } else if (tdsValue > 150 && tdsValue <= 500) {
    digitalWrite(ledYellowTDS, HIGH);  // Yellow ON - Acceptable (150-500 ppm)
  } else {
    digitalWrite(ledRedTDS, HIGH);     // Red ON - Poor (>500 ppm)
  }
}

void updateTurbidityLEDs(float turbidityNTU) {
  // Turn off all turbidity LEDs first
  digitalWrite(ledGreenTurbidity, LOW);
  digitalWrite(ledYellowTurbidity, LOW);
  digitalWrite(ledRedTurbidity, LOW);
  
  // If no water detected, all LEDs remain off
  if (turbidityNTU < 0) {
    return;
  }
  
  // Control turbidity LEDs based on NTU value ranges
  if (turbidityNTU <= 1.0) {
    digitalWrite(ledGreenTurbidity, HIGH);   // Green ON - Excellent (0-1.0 NTU)
  } else if (turbidityNTU > 1.0 && turbidityNTU <= 5.0) {
    digitalWrite(ledYellowTurbidity, HIGH);  // Yellow ON - Acceptable (1.0-5.0 NTU)
  } else {
    digitalWrite(ledRedTurbidity, HIGH);     // Red ON - Poor (>5.0 NTU)
  }
}

void calibrateTDS(float knownTDSValue) {
  // Take a reading
  int rawAnalog;
  float voltage;
  float measuredTDS = readTDS(rawAnalog, voltage);
  
  if (measuredTDS < 0) {
    Serial.println("Cannot calibrate - sensor not in water");
    return;
  }
  
  // Store the measured value for this calibration point
  if (currentCalibrationPoint < NUM_CALIBRATION_POINTS) {
    calibrationKnownValues[currentCalibrationPoint] = knownTDSValue;
    calibrationMeasuredValues[currentCalibrationPoint] = measuredTDS;
    currentCalibrationPoint++;
    
    Serial.printf("Calibration point %d recorded: Known=%.2f, Measured=%.2f\n", 
                  currentCalibrationPoint, knownTDSValue, measuredTDS);
    
    // If we have all points, calculate the calibration
    if (currentCalibrationPoint >= NUM_CALIBRATION_POINTS) {
      calculateCalibration();
    } else {
      Serial.printf("Please add calibration solution with TDS %.2f ppm\n", 
                    calibrationKnownValues[currentCalibrationPoint]);
    }
  }
}

void calculateCalibration() {
  // Calculate linear regression for better calibration (y = mx + b)
  float sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0;
  int n = NUM_CALIBRATION_POINTS;
  
  for (int i = 0; i < n; i++) {
    sumX += calibrationMeasuredValues[i];
    sumY += calibrationKnownValues[i];
    sumXY += calibrationMeasuredValues[i] * calibrationKnownValues[i];
    sumX2 += calibrationMeasuredValues[i] * calibrationMeasuredValues[i];
  }
  
  // Calculate slope (m) and intercept (b)
  calibrationFactor = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
  calibrationOffset = (sumY - calibrationFactor * sumX) / n;
  
  // Save to preferences
  preferences.putFloat("factor", calibrationFactor);
  preferences.putFloat("offset", calibrationOffset);
  
  Serial.printf("Calibration complete. Factor: %.3f, Offset: %.2f\n", 
                calibrationFactor, calibrationOffset);
  
  // Reset for next calibration
  currentCalibrationPoint = 0;
}

void printWaterQualityInfo(float tdsValue, float turbidityNTU) {
  Serial.println("\n=== WATER QUALITY ASSESSMENT ===");
  
  // TDS assessment
  Serial.print("TDS: ");
  if (tdsValue < 0) {
    Serial.println("No water detected");
  } else if (tdsValue <= 50) {
    Serial.println("Excellent (0-50 ppm) - Very pure water");
  } else if (tdsValue <= 150) {
    Serial.println("Good (50-150 ppm) - Fresh, natural taste");
  } else if (tdsValue <= 300) {
    Serial.println("Acceptable (150-300 ppm) - Normal for groundwater/tap");
  } else if (tdsValue <= 500) {
    Serial.println("Moderate (300-500 ppm) - Upper WHO safe limit");
  } else if (tdsValue <= 1200) {
    Serial.println("Poor (500-1200 ppm) - Not recommended for drinking");
  } else {
    Serial.println("Very Bad (>1200 ppm) - Unsafe for consumption");
  }
  
  // Turbidity assessment
  Serial.print("Turbidity: ");
  if (turbidityNTU < 0) {
    Serial.println("No water detected");
  } else if (turbidityNTU <= 1.0) {
    Serial.println("Excellent (0.0-1.0 NTU) - Very clear water (WHO standard)");
  } else if (turbidityNTU <= 5.0) {
    Serial.println("Acceptable (1-5 NTU) - Still safe for drinking");
  } else if (turbidityNTU <= 50.0) {
    Serial.println("Concerning (>5 NTU) - Not recommended for drinking");
  } else {
    Serial.println("Unsafe (>50 NTU) - Very turbid, likely polluted");
  }
  Serial.println("================================");
}

void loop() {
  unsigned long currentTime = millis();
  
  // Take reading at regular intervals
  if (currentTime - lastReadingTime >= READING_INTERVAL || lastReadingTime == 0) {
    lastReadingTime = currentTime;
    
    // Read TDS value with raw analog and voltage references
    int tdsRawAnalog;
    float tdsVoltage;
    float tdsValue = readTDS(tdsRawAnalog, tdsVoltage);
    
    // Read turbidity with all parameters
    int turbidityRawADC;
    float voutESP32, sensorVoltage;
    float turbidityNTU = readTurbidity(turbidityRawADC, voutESP32, sensorVoltage);
    
    // Update TDS LEDs based on TDS reading
    updateTDSLEDs(tdsValue);
    
    // Update turbidity LEDs based on turbidity reading
    updateTurbidityLEDs(turbidityNTU);
    
    // Display readings in clean format
    Serial.println("\n=== SENSOR READINGS ===");
    
    // TDS readings
    Serial.println("TDS SENSOR:");
    if (tdsValue >= 0) {
      Serial.printf("  TDS Value: %.2f ppm\n", tdsValue);
      Serial.printf("  ADC Reading: %d\n", tdsRawAnalog);
      Serial.printf("  Voltage: %.3f V\n", tdsVoltage);
      
      // LED status for TDS
      if (tdsValue <= 150) {
        Serial.println("  Status: Excellent (Green TDS LED)");
      } else if (tdsValue > 150 && tdsValue <= 500) {
        Serial.println("  Status: Acceptable (Yellow TDS LED)");
      } else {
        Serial.println("  Status: Poor (Red TDS LED)");
      }
    } else {
      Serial.println("  TDS Value: No water detected");
      Serial.printf("  ADC Reading: %d\n", tdsRawAnalog);
      Serial.printf("  Voltage: %.3f V\n", tdsVoltage);
      Serial.println("  Status: No water detected (No TDS LEDs)");
    }
    
    // Turbidity readings
    Serial.println("TURBIDITY SENSOR:");
    if (turbidityNTU >= 0) {
      Serial.printf("  Turbidity Value: %.1f NTU\n", turbidityNTU);
      Serial.printf("  Analog Value: %d\n", turbidityRawADC);
      Serial.printf("  Sensor Voltage: %.3f V\n", sensorVoltage);
      
      // LED status for turbidity
      if (turbidityNTU <= 1.0) {
        Serial.println("  Status: Excellent (Green Turbidity LED)");
      } else if (turbidityNTU > 1.0 && turbidityNTU <= 5.0) {
        Serial.println("  Status: Acceptable (Yellow Turbidity LED)");
      } else {
        Serial.println("  Status: Poor (Red Turbidity LED)");
      }
    } else {
      Serial.println("  Turbidity Value: No water detected");
      Serial.printf("  Analog Value: %d\n", turbidityRawADC);
      Serial.printf("  Sensor Voltage: %.3f V\n", sensorVoltage);
      Serial.println("  Status: No water detected (No Turbidity LEDs)");
    }
    
    Serial.println("=======================");
    
    // Print water quality assessment
    printWaterQualityInfo(tdsValue, turbidityNTU);
    
    // Always send data to server if connected (let server decide what to do with dry readings)
    if (WiFi.status() == WL_CONNECTED) {
      Serial.println("Sending data to server...");
      if (sendDataToServer(tdsValue, tdsRawAnalog, tdsVoltage, 
                          turbidityNTU, turbidityRawADC, voutESP32, sensorVoltage)) {
        Serial.println("Data sent successfully");
      } else {
        Serial.println("Failed to send data");
      }
    } else {
      Serial.println("WiFi not connected - storing data locally");
      // Here you could implement local storage (SD card, EEPROM, etc.)
    }
    
    // Reset dry count after processing
    if (dryCount >= CONSECUTIVE_DRY_READINGS) {
      Serial.println("Note: TDS sensor appears to be out of water");
      dryCount = 0; // Reset counter
    }
  }
  
  // Handle calibration command via serial
  if (Serial.available()) {
    String command = Serial.readStringUntil('\n');
    command.trim();
    
    if (command.startsWith("CAL:")) {
      float knownValue = command.substring(4).toFloat();
      if (knownValue > 0) {
        Serial.printf("Starting calibration with known value: %.2f ppm\n", knownValue);
        calibrateTDS(knownValue);
      }
    } else if (command == "MULTI_CAL") {
      currentCalibrationPoint = 0;
      Serial.println("Starting multi-point calibration");
      Serial.println("Please prepare solutions with known TDS values:");
      Serial.println("1. 0 ppm (distilled water)");
      Serial.println("2. 500 ppm");
      Serial.println("3. 1000 ppm");
      Serial.println("Enter CAL:0 for the first calibration point");
    } else if (command == "RESET_CAL") {
      calibrationFactor = 1.0;
      calibrationOffset = 0.0;
      preferences.putFloat("factor", 1.0);
      preferences.putFloat("offset", 0.0);
      Serial.println("Calibration reset to default");
    } else if (command == "MEASURE_VREF") {
      measureVref();
      Serial.printf("Measured VREF: %.3fV\n", measuredVref);
    } else if (command == "STATUS") {
      Serial.printf("Calibration factor: %.3f\n", calibrationFactor);
      Serial.printf("Calibration offset: %.2f\n", calibrationOffset);
      Serial.printf("Measured VREF: %.3fV\n", measuredVref);
      Serial.printf("WiFi status: %s\n", WiFi.status() == WL_CONNECTED ? "Connected" : "Disconnected");
      Serial.printf("Dry count: %d/%d\n", dryCount, CONSECUTIVE_DRY_READINGS);
    } else if (command == "THRESHOLDS") {
      Serial.printf("Dry sensor threshold: %d\n", DRY_SENSOR_THRESHOLD);
      Serial.printf("Minimum voltage: %.2fV\n", MIN_VOLTAGE);
      Serial.printf("Consecutive dry readings: %d\n", CONSECUTIVE_DRY_READINGS);
    } else if (command == "SET_DEVICE_ID") {
      Serial.println("Enter new device ID:");
      while (!Serial.available()) {
        delay(100);
      }
      deviceId = Serial.readStringUntil('\n');
      deviceId.trim();
      Serial.println("Device ID set to: " + deviceId);
    } else if (command == "TEST_SENSORS") {
      testSensorConnections();
    } else if (command == "LED_TEST") {
      Serial.println("Testing all LEDs...");
      // Test TDS LEDs
      digitalWrite(ledGreenTDS, HIGH); delay(500); digitalWrite(ledGreenTDS, LOW);
      digitalWrite(ledYellowTDS, HIGH); delay(500); digitalWrite(ledYellowTDS, LOW);
      digitalWrite(ledRedTDS, HIGH); delay(500); digitalWrite(ledRedTDS, LOW);
      // Test turbidity LEDs
      digitalWrite(ledGreenTurbidity, HIGH); delay(500); digitalWrite(ledGreenTurbidity, LOW);
      digitalWrite(ledYellowTurbidity, HIGH); delay(500); digitalWrite(ledYellowTurbidity, LOW);
      digitalWrite(ledRedTurbidity, HIGH); delay(500); digitalWrite(ledRedTurbidity, LOW);
      Serial.println("LED test complete");
    }
  }
  
  // Small delay to prevent busy waiting
  delay(100);
}