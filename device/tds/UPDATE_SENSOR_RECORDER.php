<?php
/**
 * UPDATE SENSOR RECEIVER TO USE SYSTEM_SENSOR ACCOUNT
 * 
 * This script updates tds_receiver.php to attribute sensor readings
 * to the system_sensor user account instead of anonymous recording.
 * 
 * Run this ONCE to update your tds_receiver.php file.
 */

require_once '../../../db.php';

// Get system_sensor user ID (it's a field_worker role)
$sensor_user_query = "SELECT id FROM users WHERE username = 'system_sensor' AND role = 'field_worker' LIMIT 1";
$result = $conn->query($sensor_user_query);

if ($result->num_rows === 0) {
    echo "❌ ERROR: system_sensor account not found!\n";
    echo "Please run SETUP_SYSTEM_SENSOR.sql first.\n";
    exit(1);
}

$sensor_user = $result->fetch_assoc();
$sensor_user_id = $sensor_user['id'];

echo "✅ Found system_sensor account (ID: $sensor_user_id, Role: field_worker)\n\n";

// Check if water_quality_reports table has recorded_by column
$check_column = $conn->query("SHOW COLUMNS FROM water_quality_reports LIKE 'recorded_by'");

if ($check_column->num_rows === 0) {
    echo "⚠️  Adding 'recorded_by' column to water_quality_reports table...\n";
    
    $add_column = "ALTER TABLE water_quality_reports 
                   ADD COLUMN recorded_by INT NULL AFTER client_id,
                   ADD CONSTRAINT fk_recorded_by FOREIGN KEY (recorded_by) REFERENCES users(id) ON DELETE SET NULL";
    
    if ($conn->query($add_column)) {
        echo "✅ Column added successfully!\n\n";
    } else {
        echo "❌ Error adding column: " . $conn->error . "\n";
        exit(1);
    }
} else {
    echo "✅ Column 'recorded_by' already exists in water_quality_reports\n\n";
}

echo "📊 Configuration Summary:\n";
echo "========================\n";
echo "System Sensor User ID: $sensor_user_id\n";
echo "Username: system_sensor\n";
echo "Role: field_worker (automated system)\n";
echo "\n";
echo "✅ Setup complete! Your sensor data will now be attributed to the system_sensor account.\n";
echo "\n";
echo "Next steps:\n";
echo "1. Update tds_receiver.php to use sensor_user_id when saving data\n";
echo "2. Create a simple dashboard for system_sensor login (optional)\n";
echo "3. Change default password for system_sensor account\n";

$conn->close();
?>

