<?php
header('Content-Type: application/json');
header('Access-Control-Allow-Origin: *');

// Database configuration
$servername = "localhost";
$username = "u520834156_userBCWaters25";
$password = "^u4ctM3J8!w";
$dbname = "u520834156_DBBagoWaters25";

// Create connection
$conn = new mysqli($servername, $username, $password, $dbname);

// Check connection
if ($conn->connect_error) {
    die(json_encode(['error' => 'Database connection failed']));
}

// Get latest turbidity readings
$result = $conn->query("SELECT * FROM turbidity_readings ORDER BY reading_time DESC LIMIT 100");
$readings = [];
if ($result->num_rows > 0) {
    while($row = $result->fetch_assoc()) {
        $readings[] = $row;
    }
}

// Get stats
$stats = $conn->query("SELECT 
    AVG(ntu_value) as avg_ntu,
    MAX(ntu_value) as max_ntu,
    MIN(ntu_value) as min_ntu,
    COUNT(*) as total_readings
    FROM turbidity_readings")->fetch_assoc();

$conn->close();

echo json_encode([
    'readings' => $readings,
    'stats' => $stats
]);
?>