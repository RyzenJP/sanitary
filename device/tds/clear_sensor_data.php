<?php
// Script to clear all sensor test data and start fresh
// WARNING: This will delete ALL sensor readings from the database!

header('Content-Type: application/json');

// Database configuration
$servername = "localhost";
$username = "u520834156_userBCWaters25";
$password = "^u4ctM3J8!w";
$dbname = "u520834156_DBBagoWaters25";

// Create connection
$conn = new mysqli($servername, $username, $password, $dbname);

// Check connection
if ($conn->connect_error) {
    die(json_encode(['success' => false, 'error' => 'Database connection failed']));
}

$results = [];

// Clear TDS readings
$query1 = "DELETE FROM tds_readings";
if ($conn->query($query1)) {
    $results['tds_deleted'] = $conn->affected_rows;
} else {
    $results['tds_error'] = $conn->error;
}

// Clear Turbidity readings
$query2 = "DELETE FROM turbidity_readings";
if ($conn->query($query2)) {
    $results['turbidity_deleted'] = $conn->affected_rows;
} else {
    $results['turbidity_error'] = $conn->error;
}

// Reset auto-increment IDs (optional)
$conn->query("ALTER TABLE tds_readings AUTO_INCREMENT = 1");
$conn->query("ALTER TABLE turbidity_readings AUTO_INCREMENT = 1");

$conn->close();

$results['success'] = true;
$results['message'] = 'All sensor test data has been cleared successfully';

echo json_encode($results, JSON_PRETTY_PRINT);
?>

