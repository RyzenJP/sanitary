<?php
/**
 * PHP Bridge to Python ML Server
 * Connects to the local Python server for AI predictions
 */

header('Content-Type: application/json');
header('Access-Control-Allow-Origin: *');
header('Access-Control-Allow-Methods: GET, POST');
header('Access-Control-Allow-Headers: Content-Type');

// Python server configuration - AUTO-DETECTS localhost vs online
// Auto-detect environment
if ($_SERVER['HTTP_HOST'] === 'localhost' || strpos($_SERVER['HTTP_HOST'], '127.0.0.1') !== false) {
    // Running on localhost - use local ML server
    $python_server_url = 'http://localhost:5000';
} else {
    // Running on online domain - update this to your school server endpoint
    $python_server_url = 'https://your-school-endpoint.example.com';
}

// Function to call Python server
function callPythonServer($endpoint, $data = []) {
    global $python_server_url;
    
    $url = $python_server_url . $endpoint;
    
    if (!empty($data)) {
        $url .= '?' . http_build_query($data);
    }
    
    $context = stream_context_create([
        'http' => [
            'timeout' => 10,
            'method' => 'GET'
        ]
    ]);
    
    $response = file_get_contents($url, false, $context);
    
    if ($response === false) {
        return [
            'status' => 'error',
            'message' => 'Python server not responding. Please start the ML server first.',
            'server_url' => $url
        ];
    }
    
    $data = json_decode($response, true);
    return $data ?: ['status' => 'error', 'message' => 'Invalid response from Python server'];
}


// Main API logic
try {
    $method = $_SERVER['REQUEST_METHOD'];
    
    if ($method === 'GET') {
        // Get parameters from URL
        $tds_value = floatval($_GET['tds'] ?? 350);
        $turbidity_value = floatval($_GET['turbidity'] ?? 0.8);
        $temperature = floatval($_GET['temperature'] ?? 25);
        $ph_level = floatval($_GET['ph'] ?? 7.0);
        $use_sensor = $_GET['use_sensor'] ?? 'false';
        
        // If using sensor data, get from database first
        if ($use_sensor === 'true') {
            // Include database connection
            require_once '../../db.php';
            
            try {
                // Get latest TDS reading
                $tds_query = "SELECT tds_value FROM tds_readings ORDER BY reading_time DESC LIMIT 1";
                $tds_result = $conn->query($tds_query);
                if ($tds_result && $tds_row = $tds_result->fetch_assoc()) {
                    $tds_value = floatval($tds_row['tds_value']);
                }
                
                // Get latest Turbidity reading
                $turbidity_query = "SELECT ntu_value FROM turbidity_readings ORDER BY reading_time DESC LIMIT 1";
                $turbidity_result = $conn->query($turbidity_query);
                if ($turbidity_result && $turbidity_row = $turbidity_result->fetch_assoc()) {
                    $turbidity_value = floatval($turbidity_row['ntu_value']);
                }
            } catch (Exception $e) {
                // Use default values if database fails
            }
        }
        
        // Call Python server
        $result = callPythonServer('/predict', [
            'tds' => $tds_value,
            'turbidity' => $turbidity_value,
            'temperature' => $temperature,
            'ph' => $ph_level
        ]);
        
        echo json_encode($result, JSON_PRETTY_PRINT);
        
    } elseif ($method === 'POST') {
        // Get parameters from POST data
        $input = json_decode(file_get_contents('php://input'), true);
        
        $tds_value = floatval($input['tds_value'] ?? 350);
        $turbidity_value = floatval($input['turbidity_value'] ?? 0.8);
        $temperature = floatval($input['temperature'] ?? 25);
        $ph_level = floatval($input['ph_level'] ?? 7.0);
        
        // Call Python server
        $result = callPythonServer('/predict', [
            'tds' => $tds_value,
            'turbidity' => $turbidity_value,
            'temperature' => $temperature,
            'ph' => $ph_level
        ]);
        
        echo json_encode($result, JSON_PRETTY_PRINT);
        
    } else {
        http_response_code(405);
        echo json_encode([
            'status' => 'error',
            'message' => 'Method not allowed'
        ]);
    }
    
} catch (Exception $e) {
    http_response_code(500);
    echo json_encode([
        'status' => 'error',
        'message' => 'Server error: ' . $e->getMessage()
    ]);
}
?>
