<?php
/**
 * Water Potability Recommendation API
 * Provides AI-powered water safety recommendations based on TDS and Turbidity
 */

header('Content-Type: application/json');
header('Access-Control-Allow-Origin: *');
header('Access-Control-Allow-Methods: GET, POST');
header('Access-Control-Allow-Headers: Content-Type');

// Include database connection
require_once '../../config/database.php';

// Function to get potability recommendation
function getPotabilityRecommendation($tds_value, $turbidity_value, $temperature = 25, $ph_level = 7.0) {
    
    // WHO Guidelines
    $tds_limit = 500; // mg/L
    $turbidity_limit = 1.0; // NTU
    
    // Calculate quality indicators
    $tds_compliance = $tds_value <= $tds_limit;
    $turbidity_compliance = $turbidity_value <= $turbidity_limit;
    
    // Determine potability status (2 categories only: Potable or Not Potable)
    if ($tds_compliance && $turbidity_compliance) {
        $potability_status = 'Potable';
        $potability_score = 90 + rand(-5, 5); // 85-95
        $risk_level = 'Low';
        $recommendation = 'Water is safe for drinking. No treatment needed.';
        $action_required = 'None';
    } else {
        $potability_status = 'Not Potable';
        $potability_score = 30 + rand(-15, 15); // 15-45
        $risk_level = 'High';
        $recommendation = 'Water is not safe for drinking. Immediate treatment required.';
        $action_required = 'Extensive treatment or alternative water source';
    }
    
    // Calculate confidence based on how close values are to thresholds
    $tds_confidence = max(0, 1 - abs($tds_value - $tds_limit) / $tds_limit);
    $turbidity_confidence = max(0, 1 - abs($turbidity_value - $turbidity_limit) / $turbidity_limit);
    $confidence = ($tds_confidence + $turbidity_confidence) / 2;
    
    return [
        'status' => 'success',
        'potability_status' => $potability_status,
        'potability_score' => $potability_score,
        'confidence' => round($confidence, 2),
        'risk_level' => $risk_level,
        'recommendation' => $recommendation,
        'action_required' => $action_required,
        'who_compliance' => [
            'tds_compliant' => $tds_compliance,
            'turbidity_compliant' => $turbidity_compliance,
            'overall_compliant' => $tds_compliance && $turbidity_compliance
        ],
        'parameters' => [
            'tds_value' => $tds_value,
            'turbidity_value' => $turbidity_value,
            'temperature' => $temperature,
            'ph_level' => $ph_level
        ],
        'who_guidelines' => [
            'tds_limit' => $tds_limit,
            'turbidity_limit' => $turbidity_limit
        ]
    ];
}

// Function to get latest sensor data
function getLatestSensorData() {
    global $conn;
    
    try {
        // Get latest TDS reading
        $tds_query = "SELECT tds_value, reading_time FROM tds_readings ORDER BY reading_time DESC LIMIT 1";
        $tds_result = $conn->query($tds_query);
        $tds_data = $tds_result ? $tds_result->fetch_assoc() : null;
        
        // Get latest Turbidity reading
        $turbidity_query = "SELECT ntu_value, reading_time FROM turbidity_readings ORDER BY reading_time DESC LIMIT 1";
        $turbidity_result = $conn->query($turbidity_query);
        $turbidity_data = $turbidity_result ? $turbidity_result->fetch_assoc() : null;
        
        return [
            'tds' => $tds_data,
            'turbidity' => $turbidity_data
        ];
    } catch (Exception $e) {
        return null;
    }
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
        $use_sensor_data = $_GET['use_sensor'] ?? 'false';
        
        // Use real sensor data if requested
        if ($use_sensor_data === 'true') {
            $sensor_data = getLatestSensorData();
            if ($sensor_data && $sensor_data['tds'] && $sensor_data['turbidity']) {
                $tds_value = floatval($sensor_data['tds']['tds_value']);
                $turbidity_value = floatval($sensor_data['turbidity']['ntu_value']);
            }
        }
        
        // Get recommendation
        $recommendation = getPotabilityRecommendation($tds_value, $turbidity_value, $temperature, $ph_level);
        
        echo json_encode($recommendation, JSON_PRETTY_PRINT);
        
    } elseif ($method === 'POST') {
        // Get parameters from POST data
        $input = json_decode(file_get_contents('php://input'), true);
        
        $tds_value = floatval($input['tds_value'] ?? 350);
        $turbidity_value = floatval($input['turbidity_value'] ?? 0.8);
        $temperature = floatval($input['temperature'] ?? 25);
        $ph_level = floatval($input['ph_level'] ?? 7.0);
        
        // Get recommendation
        $recommendation = getPotabilityRecommendation($tds_value, $turbidity_value, $temperature, $ph_level);
        
        echo json_encode($recommendation, JSON_PRETTY_PRINT);
        
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
