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

// Get latest readings (order by ID to get absolute latest)
$result = $conn->query("SELECT * FROM tds_readings ORDER BY id DESC LIMIT 100");
$readings = [];
if ($result->num_rows > 0) {
    while($row = $result->fetch_assoc()) {
        $readings[] = $row;
    }
}

// Get stats
$stats = $conn->query("SELECT 
    AVG(tds_value) as avg_tds,
    MAX(tds_value) as max_tds,
    MIN(tds_value) as min_tds,
    COUNT(*) as total_readings
    FROM tds_readings")->fetch_assoc();

$conn->close();

echo json_encode([
    'readings' => $readings,
    'stats' => $stats
]);
?>