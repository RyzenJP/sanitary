<?php
header('Content-Type: application/json');
header('Access-Control-Allow-Origin: *');
header('Access-Control-Allow-Methods: GET, POST, OPTIONS');
header('Access-Control-Allow-Headers: Content-Type');

// Handle preflight request
if ($_SERVER['REQUEST_METHOD'] === 'OPTIONS') {
    http_response_code(200);
    exit();
}

// Database configuration
$servername = "localhost";
$username = "u520834156_userBCWaters25";
$password = "^u4ctM3J8!w";
$dbname = "u520834156_DBBagoWaters25";

// Create connection
$conn = new mysqli($servername, $username, $password, $dbname);

// Check connection
if ($conn->connect_error) {
    http_response_code(500);
    echo json_encode([
        'status' => 'error',
        'message' => 'Database connection failed: ' . $conn->connect_error
    ]);
    exit();
}

// Get request data (works with both GET and POST)
$request_data = [];
if ($_SERVER['REQUEST_METHOD'] === 'POST') {
    $content_type = isset($_SERVER['CONTENT_TYPE']) ? $_SERVER['CONTENT_TYPE'] : '';
    
    if (strpos($content_type, 'application/json') !== false) {
        $json = file_get_contents('php://input');
        $request_data = json_decode($json, true);
    } else if (strpos($content_type, 'application/x-www-form-urlencoded') !== false) {
        parse_str(file_get_contents("php://input"), $request_data);
    } else {
        $request_data = $_POST;
    }
} else {
    $request_data = $_GET;
}

// Log received data for debugging
file_put_contents('sensor_log.txt', date('Y-m-d H:i:s') . " - " . print_r($request_data, true) . "\n", FILE_APPEND);

// Check for required parameters
$required_params = ['device_id'];
$missing_params = [];

foreach ($required_params as $param) {
    if (!isset($request_data[$param]) || $request_data[$param] === '') {
        $missing_params[] = $param;
    }
}

if (!empty($missing_params)) {
    http_response_code(400);
    echo json_encode([
        'status' => 'error',
        'message' => 'Missing required parameters: ' . implode(', ', $missing_params),
        'received_data' => $request_data
    ]);
    exit();
}

$device_id = $conn->real_escape_string($request_data['device_id']);

// Check if we have TDS data
if (isset($request_data['tds_value'])) {
    // Prepare TDS data
    $tds_value = floatval($request_data['tds_value']);
    $analog_value = intval($request_data['analog_value']);
    $voltage = floatval($request_data['voltage']);
    
    // Insert into TDS table
    $sql = "INSERT INTO tds_readings (tds_value, analog_value, voltage, reading_time) VALUES (?, ?, ?, NOW())";
    $stmt = $conn->prepare($sql);
    
    if (!$stmt) {
        http_response_code(500);
        echo json_encode([
            'status' => 'error',
            'message' => 'Prepare failed for TDS: ' . $conn->error
        ]);
        exit();
    }
    
    $stmt->bind_param("did", $tds_value, $analog_value, $voltage);
    
    if (!$stmt->execute()) {
        http_response_code(500);
        echo json_encode([
            'status' => 'error',
            'message' => 'Execute failed for TDS: ' . $stmt->error
        ]);
        $stmt->close();
        exit();
    }
    
    $stmt->close();
}

// Check if we have turbidity data
if (isset($request_data['turbidity'])) {
    // Prepare turbidity data
    $ntu_value = floatval($request_data['turbidity']);
    $turbidity_analog = isset($request_data['turbidity_raw_adc']) ? intval($request_data['turbidity_raw_adc']) : 0;
    $turbidity_voltage = isset($request_data['turbidity_sensor_voltage']) ? floatval($request_data['turbidity_sensor_voltage']) : 0;
    $vout_esp32 = isset($request_data['turbidity_vout_esp32']) ? floatval($request_data['turbidity_vout_esp32']) : 0;
    
    // Determine water quality based on NTU value
    $water_quality = 'bad'; // default
    if ($ntu_value <= 1.0) {
        $water_quality = 'good';
    } else if ($ntu_value > 1.0 && $ntu_value <= 50.0) {
        $water_quality = 'warning';
    }

    // Insert into turbidity table
    $sql = "INSERT INTO turbidity_readings (ntu_value, analog_value, voltage, raw_adc, vout_esp32, sensor_voltage, water_quality) VALUES (?, ?, ?, ?, ?, ?, ?)";
    $stmt = $conn->prepare($sql);
    
    if (!$stmt) {
        http_response_code(500);
        echo json_encode([
            'status' => 'error',
            'message' => 'Prepare failed for turbidity: ' . $conn->error
        ]);
        exit();
    }
    
    $stmt->bind_param("dddddds", $ntu_value, $turbidity_analog, $turbidity_voltage, $turbidity_analog, $vout_esp32, $turbidity_voltage, $water_quality);
    
    if (!$stmt->execute()) {
        http_response_code(500);
        echo json_encode([
            'status' => 'error',
            'message' => 'Execute failed for turbidity: ' . $stmt->error
        ]);
        $stmt->close();
        exit();
    }
    
    $stmt->close();
}

// If we reached here, both inserts were successful
echo json_encode([
    'status' => 'success',
    'message' => 'Data saved successfully',
    'data' => [
        'device_id' => $device_id,
        'timestamp' => date('Y-m-d H:i:s')
    ]
]);

$conn->close();
?>