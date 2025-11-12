# 🔧 Turbidity Data Not Saving - Fix Guide

## 🔍 **Problems Identified in Your Sketch**

I analyzed your Arduino sketch and found **3 critical issues** preventing Turbidity data from saving:

---

## ❌ **Issue 1: Server URL Points to Wrong Location**

### **Current Code (Line 36):**
```cpp
char serverUrl[150] = "https://192.168.137.198/sanitary1/sanitary/device/tds/tds_receiver.php";
```

### **Problem:**
The URL has **`https://`** but your local server likely uses **`http://`** (no SSL)

### **Fix:**
```cpp
char serverUrl[150] = "http://192.168.137.198/sanitary/sensor/device/tds/tds_receiver.php";
```

**Changes:**
1. ✅ Changed `https://` to `http://`
2. ✅ Changed `/sanitary1/sanitary/` to `/sanitary/` (match your actual path)
3. ✅ Changed `/device/tds/` to `/sensor/device/tds/` (correct folder)

---

## ❌ **Issue 2: Turbidity Sensor Returns -1 When No Water**

### **Current Code (Line 324-326):**
```cpp
// Check if sensor is in water (similar to TDS sensor)
if (rawADC < DRY_SENSOR_THRESHOLD || sensorVoltage < MIN_VOLTAGE) {
    return -1.0; // Special value indicating no water
}
```

### **Problem:**
When testing in air (not in water), turbidity sensor returns **-1**, which causes the server to receive **invalid data**.

### **How It Affects Data:**
```
Turbidity in air → rawADC < 100 → returns -1.0 NTU → Server receives -1.0 → Invalid!
```

### **Fix Option 1: Disable Water Check for Testing**
```cpp
// COMMENT OUT water check for testing
// if (rawADC < DRY_SENSOR_THRESHOLD || sensorVoltage < MIN_VOLTAGE) {
//     return -1.0;
// }

// OR set it to return 0 instead:
if (rawADC < DRY_SENSOR_THRESHOLD || sensorVoltage < MIN_VOLTAGE) {
    return 0.0; // Return 0 NTU for no water (valid value)
}
```

### **Fix Option 2: Adjust Threshold for Air Testing**
```cpp
// Lower threshold for air testing
const int DRY_SENSOR_THRESHOLD = 10;  // Was 100, now 10
```

---

## ❌ **Issue 3: PHP Server INSERT Statement Has Wrong Columns**

### **Current PHP Code (Line 126):**
```php
$sql = "INSERT INTO turbidity_readings (ntu_value, analog_value, voltage, raw_adc, vout_esp32, sensor_voltage, water_quality) VALUES (?, ?, ?, ?, ?, ?, ?)";
```

### **Problem:**
The table might not have all these columns, causing SQL errors.

### **Check Your Database:**
Run this SQL to see your table structure:
```sql
DESCRIBE turbidity_readings;
```

### **Expected Columns:**
- `id` (auto-increment)
- `ntu_value` (float)
- `analog_value` (int)
- `voltage` (float)
- `reading_time` (datetime)
- `raw_adc` (int) - **might be missing**
- `vout_esp32` (float) - **might be missing**
- `sensor_voltage` (float) - **might be missing**
- `water_quality` (varchar) - **might be missing**

---

## ✅ **COMPLETE FIX - Updated Arduino Sketch**

Replace lines 36, 100, and 324 in your sketch:

### **Fix 1: Update Server URL (Line 36)**
```cpp
// OLD:
// char serverUrl[150] = "https://192.168.137.198/sanitary1/sanitary/device/tds/tds_receiver.php";

// NEW:
char serverUrl[150] = "http://192.168.137.198/sanitary/sensor/device/tds/tds_receiver.php";
```

### **Fix 2: Update Dry Sensor Threshold (Line 26)**
```cpp
// OLD:
// const int DRY_SENSOR_THRESHOLD = 100;

// NEW (for testing without water):
const int DRY_SENSOR_THRESHOLD = 10;  // Lower threshold for air testing
```

### **Fix 3: Return 0 Instead of -1 for No Water (Line 324-326)**
```cpp
// OLD:
// if (rawADC < DRY_SENSOR_THRESHOLD || sensorVoltage < MIN_VOLTAGE) {
//     return -1.0; // Special value indicating no water
// }

// NEW:
if (rawADC < DRY_SENSOR_THRESHOLD || sensorVoltage < MIN_VOLTAGE) {
    return 0.0; // Return 0 NTU for dry sensor (valid for testing)
}
```

---

## 📊 **Update PHP Receiver to Handle Missing Columns**

Update `sensor/device/tds/tds_receiver.php` line 126:

### **Simplified INSERT (if columns are missing):**
```php
// OLD:
// $sql = "INSERT INTO turbidity_readings (ntu_value, analog_value, voltage, raw_adc, vout_esp32, sensor_voltage, water_quality) VALUES (?, ?, ?, ?, ?, ?, ?)";

// NEW (basic columns only):
$sql = "INSERT INTO turbidity_readings (ntu_value, analog_value, voltage, reading_time) VALUES (?, ?, ?, NOW())";
$stmt = $conn->prepare($sql);
$stmt->bind_param("ddd", $ntu_value, $turbidity_analog, $turbidity_voltage);
```

---

## 🗄️ **Create Turbidity Table (If Missing)**

Run this SQL in phpMyAdmin:

```sql
CREATE TABLE IF NOT EXISTS `turbidity_readings` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `ntu_value` float NOT NULL,
  `analog_value` int(11) NOT NULL,
  `voltage` float NOT NULL,
  `raw_adc` int(11) DEFAULT NULL,
  `vout_esp32` float DEFAULT NULL,
  `sensor_voltage` float DEFAULT NULL,
  `water_quality` varchar(20) DEFAULT NULL,
  `reading_time` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  KEY `reading_time` (`reading_time`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
```

---

## 🧪 **Testing Steps**

### **Step 1: Upload Fixed Sketch**
1. Update the 3 lines in your Arduino sketch
2. Upload to ESP32
3. Open Serial Monitor (115200 baud)

### **Step 2: Watch Serial Output**
You should see:
```
=== SENSOR READINGS ===
TURBIDITY SENSOR:
  Turbidity Value: 2.5 NTU
  Analog Value: 1234
  Sensor Voltage: 2.456 V
  Status: Acceptable (Yellow Turbidity LED)

Sending data to server...
HTTP response code: 200
Server response: {"status":"success"...}
Data sent successfully
```

### **Step 3: Check Database**
```sql
SELECT * FROM turbidity_readings ORDER BY id DESC LIMIT 10;
```

You should see new records!

### **Step 4: Check Sensor Log**
Check: `sensor/device/tds/sensor_log.txt`
```
2025-10-17 04:30:15 - Array
(
    [tds_value] => 236.10
    [analog_value] => 590
    [voltage] => 3.300
    [turbidity] => 2.5      ← Should see this!
    [turbidity_raw_adc] => 1234
    [turbidity_vout_esp32] => 1.234
    [turbidity_sensor_voltage] => 2.456
    [device_id] => SENSOR_001
)
```

---

## 🔍 **Debug Commands**

### **In Serial Monitor, type:**

**Check sensor readings:**
```
TEST_SENSORS
```

**Check current status:**
```
STATUS
```

**Test LEDs:**
```
LED_TEST
```

**See thresholds:**
```
THRESHOLDS
```

---

## 🚨 **Common Issues & Solutions**

### **Issue: "HTTP begin failed"**
**Solution:** Check server URL, ensure XAMPP is running

### **Issue: "HTTP response code: 404"**
**Solution:** URL path is wrong, verify file location

### **Issue: "HTTP response code: 500"**
**Solution:** PHP error, check `sensor_log.txt` for details

### **Issue: "Turbidity Value: No water detected"**
**Solution:** Threshold too high, lower `DRY_SENSOR_THRESHOLD`

### **Issue: "Failed to send data"**
**Solution:** 
- Check WiFi connection
- Verify server URL
- Check if XAMPP Apache is running

---

## ✅ **Quick Fix Summary**

1. **Change URL:**
   ```cpp
   "http://192.168.137.198/sanitary/sensor/device/tds/tds_receiver.php"
   ```

2. **Lower threshold:**
   ```cpp
   const int DRY_SENSOR_THRESHOLD = 10;
   ```

3. **Return 0 instead of -1:**
   ```cpp
   return 0.0; // instead of return -1.0;
   ```

4. **Verify table exists** in database

5. **Test and check `sensor_log.txt`**

---

## 📝 **Expected Working Output**

```
=== SENSOR READINGS ===
TDS SENSOR:
  TDS Value: 236.10 ppm
  ADC Reading: 590
  Voltage: 3.300 V
  Status: Acceptable (Yellow TDS LED)

TURBIDITY SENSOR:
  Turbidity Value: 2.5 NTU
  Analog Value: 1234
  Sensor Voltage: 2.456 V
  Status: Acceptable (Yellow Turbidity LED)

=======================

Sending data to server...
HTTP response code: 200
Server response: {"status":"success","message":"Data saved successfully"}
Data sent successfully
```

---

## 🎯 **For Your Defense Tomorrow**

If asked about sensor integration:

> "Our IoT system uses ESP32 microcontrollers to read TDS and Turbidity sensors. The device takes 50 samples for TDS and 20 for Turbidity, uses median filtering to reduce noise, and sends data every 30 seconds via HTTP POST to our PHP API, which stores it in MySQL for ML processing."

**Good luck! 🚀**

