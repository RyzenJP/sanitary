<?php
/**
 * Test script to simulate ESP32 sending data to tds_receiver.php
 * Run this in your browser to verify the receiver is working
 */

echo "<h2>🧪 Testing TDS Receiver</h2>";

// Simulate ESP32 POST data
$test_data = [
    'tds_value' => 236.12,
    'analog_value' => 1205,
    'voltage' => 0.969,
    'turbidity' => 2.5,
    'turbidity_raw_adc' => 1234,
    'turbidity_vout_esp32' => 0.995,
    'turbidity_sensor_voltage' => 2.985,
    'device_id' => 'SENSOR_001'
];

echo "<h3>📦 Test Data Being Sent:</h3>";
echo "<pre>";
print_r($test_data);
echo "</pre>";

// Convert to POST format
$post_data = http_build_query($test_data);

// Send to receiver
$url = 'http://localhost/sanitary/main/sensor/device/tds/tds_receiver.php';

$ch = curl_init($url);
curl_setopt($ch, CURLOPT_POST, 1);
curl_setopt($ch, CURLOPT_POSTFIELDS, $post_data);
curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
curl_setopt($ch, CURLOPT_HTTPHEADER, [
    'Content-Type: application/x-www-form-urlencoded'
]);

echo "<h3>🚀 Sending to: $url</h3>";

$response = curl_exec($ch);
$http_code = curl_getinfo($ch, CURLINFO_HTTP_CODE);
curl_close($ch);

echo "<h3>📡 HTTP Response Code: $http_code</h3>";

if ($http_code == 200) {
    echo "<p style='color: green; font-weight: bold;'>✅ SUCCESS! Server accepted the data.</p>";
} else {
    echo "<p style='color: red; font-weight: bold;'>❌ ERROR! Server returned: $http_code</p>";
}

echo "<h3>📄 Server Response:</h3>";
echo "<pre>";
echo htmlspecialchars($response);
echo "</pre>";

// Decode JSON response
$json_response = json_decode($response, true);

if ($json_response) {
    echo "<h3>🔍 Parsed Response:</h3>";
    echo "<ul>";
    echo "<li><strong>Status:</strong> " . ($json_response['status'] ?? 'N/A') . "</li>";
    echo "<li><strong>Message:</strong> " . ($json_response['message'] ?? 'N/A') . "</li>";
    if (isset($json_response['data'])) {
        echo "<li><strong>Device ID:</strong> " . ($json_response['data']['device_id'] ?? 'N/A') . "</li>";
        echo "<li><strong>Timestamp:</strong> " . ($json_response['data']['timestamp'] ?? 'N/A') . "</li>";
    }
    echo "</ul>";
}

// Check database
echo "<h3>💾 Checking Database...</h3>";

$servername = "localhost";
$username = "u520834156_userBCWaters25";
$password = "^u4ctM3J8!w";
$dbname = "u520834156_DBBagoWaters25";

$conn = new mysqli($servername, $username, $password, $dbname);

if ($conn->connect_error) {
    echo "<p style='color: red;'>❌ Database connection failed: " . $conn->connect_error . "</p>";
} else {
    echo "<p style='color: green;'>✅ Database connected successfully</p>";
    
    // Check TDS data
    $sql = "SELECT * FROM tds_readings ORDER BY id DESC LIMIT 3";
    $result = $conn->query($sql);
    
    echo "<h4>📊 Latest TDS Readings:</h4>";
    if ($result && $result->num_rows > 0) {
        echo "<table border='1' cellpadding='5' style='border-collapse: collapse;'>";
        echo "<tr><th>ID</th><th>TDS Value</th><th>Analog</th><th>Voltage</th><th>Quality</th><th>Time</th></tr>";
        while ($row = $result->fetch_assoc()) {
            echo "<tr>";
            echo "<td>" . $row['id'] . "</td>";
            echo "<td>" . $row['tds_value'] . " ppm</td>";
            echo "<td>" . $row['analog_value'] . "</td>";
            echo "<td>" . $row['voltage'] . " V</td>";
            echo "<td>" . (isset($row['water_quality']) ? $row['water_quality'] : 'N/A') . "</td>";
            echo "<td>" . $row['reading_time'] . "</td>";
            echo "</tr>";
        }
        echo "</table>";
    } else {
        echo "<p>No TDS data found.</p>";
    }
    
    // Check Turbidity data
    $sql = "SELECT * FROM turbidity_readings ORDER BY id DESC LIMIT 3";
    $result = $conn->query($sql);
    
    echo "<h4>📊 Latest Turbidity Readings:</h4>";
    if ($result && $result->num_rows > 0) {
        echo "<table border='1' cellpadding='5' style='border-collapse: collapse;'>";
        echo "<tr><th>ID</th><th>NTU Value</th><th>Analog</th><th>Voltage</th><th>Quality</th><th>Time</th></tr>";
        while ($row = $result->fetch_assoc()) {
            echo "<tr>";
            echo "<td>" . $row['id'] . "</td>";
            echo "<td>" . $row['ntu_value'] . " NTU</td>";
            echo "<td>" . $row['analog_value'] . "</td>";
            echo "<td>" . $row['voltage'] . " V</td>";
            echo "<td>" . (isset($row['water_quality']) ? $row['water_quality'] : 'N/A') . "</td>";
            echo "<td>" . $row['reading_time'] . "</td>";
            echo "</tr>";
        }
        echo "</table>";
    } else {
        echo "<p style='color: orange;'>⚠️ No turbidity data found. This might indicate turbidity data is not being received.</p>";
    }
    
    $conn->close();
}

echo "<hr>";
echo "<h3>✅ Next Steps:</h3>";
echo "<ol>";
echo "<li>If you see SUCCESS above and new database entries, the receiver is working! ✅</li>";
echo "<li>Upload the fixed sketch.ino to your ESP32</li>";
echo "<li>Open Serial Monitor (115200 baud)</li>";
echo "<li>Watch for 'Data sent successfully' messages</li>";
echo "<li>Check this page again to see real ESP32 data appearing</li>";
echo "</ol>";

echo "<p><a href='test_receiver.php'><button>🔄 Run Test Again</button></a></p>";
echo "<p><a href='sensor_log.txt' target='_blank'><button>📄 View Log File</button></a></p>";
?>

